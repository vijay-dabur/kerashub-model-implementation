import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.phi4_mini.phi4_mini_rotary_embedding import (
    Phi4MiniRotaryEmbedding,
)
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


class Phi4MiniAttention(keras.layers.Layer):
    """Grouped Query Attention with partial RoPE for Phi-4-mini.

    Phi-4-mini differs from Phi-3 attention in two key ways:

    1. **Partial Rotary Position Embedding**: Only the first
       ``floor(head_dim * partial_rotary_factor)`` dimensions of each
       query/key head are rotated.  The remaining dimensions are passed
       through unchanged.  For the production model this is
       ``floor(128 * 0.75) = 96`` out of 128 dimensions per head.

    2. **Grouped Query Attention (GQA)**: 24 query heads share 8 key-value
       heads (3 query heads per KV pair), reducing the KV-cache size by 3×.

    Args:
        num_query_heads (int): Total number of query attention heads.
        num_key_value_heads (int): Number of key-value heads.
        partial_rotary_factor (float): Fraction of head dimensions to apply
            RoPE to.  Must produce an even ``rotary_dim``.
        kernel_initializer: Weight initializer for all dense projections.
        dropout (float): Attention-weight dropout probability.
        max_sequence_length (int): Maximum sequence length.
        pretraining_sequence_length (int): Sequence length used during
            pretraining (controls SuScaling threshold).
        rope_max_wavelength (int): Base wavelength θ for standard RoPE.
        rope_scaling_type (str | None): ``None`` for vanilla RoPE, ``"su"``
            for SuScaled / LongRoPE.
        rope_scaling_short_factor (list[float] | None): Per-frequency
            divisors for sequences ≤ pretraining length.
        rope_scaling_long_factor (list[float] | None): Per-frequency
            divisors for sequences > pretraining length.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        partial_rotary_factor=0.75,
        kernel_initializer="glorot_uniform",
        dropout=0,
        max_sequence_length=131072,
        pretraining_sequence_length=4096,
        rope_max_wavelength=10000,
        rope_scaling_type=None,
        rope_scaling_short_factor=None,
        rope_scaling_long_factor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.dropout = dropout

        self.max_sequence_length = max_sequence_length
        self.pretraining_sequence_length = pretraining_sequence_length
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_short_factor = rope_scaling_short_factor
        self.rope_scaling_long_factor = rope_scaling_long_factor

        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        hidden_dim = inputs_shape[-1]
        head_dim = hidden_dim // self.num_query_heads
        self._inv_norm_factor = 1.0 / math.sqrt(head_dim)

        # Compute the number of dimensions that receive rotary encoding.
        self.rotary_dim = int(head_dim * self.partial_rotary_factor)
        if self.rotary_dim % 2 != 0:
            raise ValueError(
                f"`rotary_dim` must be even, but got {self.rotary_dim}. "
                f"Adjust `partial_rotary_factor` so that "
                f"`floor(head_dim * partial_rotary_factor)` is even. "
                f"head_dim={head_dim}, partial_rotary_factor="
                f"{self.partial_rotary_factor}."
            )

        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self.softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self.output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build((None, None, self.num_query_heads, head_dim))

        if self.rope_scaling_type is None:
            self.rotary_embedding_layer = RotaryEmbedding(
                max_wavelength=self.rope_max_wavelength,
                dtype=self.dtype_policy,
            )
        elif self.rope_scaling_type == "su":
            num_freq = self.rotary_dim // 2
            if len(self.rope_scaling_short_factor) != num_freq:
                raise ValueError(
                    "`rope_scaling_short_factor` must be of length "
                    "`rotary_dim // 2 = "
                    f"floor(head_dim * partial_rotary_factor) // 2`. "
                    f"`len(rope_scaling_short_factor)` is "
                    f"{len(self.rope_scaling_short_factor)} "
                    f"while it should be {num_freq}."
                )
            if len(self.rope_scaling_long_factor) != num_freq:
                raise ValueError(
                    "`rope_scaling_long_factor` must be of length "
                    "`rotary_dim // 2 = "
                    f"floor(head_dim * partial_rotary_factor) // 2`. "
                    f"`len(rope_scaling_long_factor)` is "
                    f"{len(self.rope_scaling_long_factor)} "
                    f"while it should be {num_freq}."
                )
            self.rotary_embedding_layer = Phi4MiniRotaryEmbedding(
                inverese_freq_short_factor=self.rope_scaling_short_factor,
                inverese_freq_long_factor=self.rope_scaling_long_factor,
                max_sequence_length=self.max_sequence_length,
                pretraining_sequence_length=self.pretraining_sequence_length,
                max_wavelength=self.rope_max_wavelength,
                dtype=self.dtype_policy,
            )
        else:
            raise ValueError(
                '`rope_scaling_type` must be `None` or `"su"`. '
                "If `None`, `RotaryEmbedding` will be used. "
                'If `"su"`, `Phi4MiniRotaryEmbedding` (SuScaled / LongRoPE) '
                "will be used."
            )

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self.query_dense(hidden_states)
        key = self.key_dense(hidden_states)
        value = self.value_dense(hidden_states)

        # --- Partial RoPE ---------------------------------------------------
        # Split query and key into rotary and pass-through portions.
        # query/key shape: [batch, seq_len, num_heads, head_dim]
        query_rot  = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        key_rot    = key[..., : self.rotary_dim]
        key_pass   = key[..., self.rotary_dim :]

        # Apply rotary embedding only to the rotary portion.
        query_rot = self.rotary_embedding_layer(
            query_rot, start_index=start_index
        )
        key_rot = self.rotary_embedding_layer(
            key_rot, start_index=start_index
        )

        # Reassemble the full head dimension.
        query = ops.concatenate([query_rot, query_pass], axis=-1)
        key   = ops.concatenate([key_rot,   key_pass],   axis=-1)
        # --------------------------------------------------------------------

        if cache is not None:
            key_cache   = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key   = key_cache
                value = value_cache
            else:
                start = [0, cache_update_index, 0, 0]
                key   = ops.slice_update(key_cache,   start, key)
                value = ops.slice_update(value_cache, start, value)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )

        # GQA: broadcast KV heads to match the number of query heads.
        # [batch, seq, num_kv_heads, head_dim] → [batch, seq, num_heads, head_dim]
        key   = ops.repeat(key,   repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query, key, value, attention_mask
        )

        attention_output = self.dropout_layer(
            attention_output, training=training
        )
        attention_output = self.output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            return self.softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        return self.softmax(attention_scores)

    def _compute_attention(self, query, key, value, attention_mask=None):
        if fused_attention_op_available():
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            attention_output = ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )
            return attention_output

        attention_scores = ops.einsum("bquh,bkuh->buqk", query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(
            "buqk,bkuh->bquh", attention_scores, value
        )
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "partial_rotary_factor": self.partial_rotary_factor,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "pretraining_sequence_length": self.pretraining_sequence_length,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_type": self.rope_scaling_type,
                "rope_scaling_short_factor": self.rope_scaling_short_factor,
                "rope_scaling_long_factor": self.rope_scaling_long_factor,
            }
        )
        return config
