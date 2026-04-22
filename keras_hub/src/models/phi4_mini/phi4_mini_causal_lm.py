from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone
from keras_hub.src.models.phi4_mini.phi4_mini_causal_lm_preprocessor import (
    Phi4MiniCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Phi4MiniCausalLM")
class Phi4MiniCausalLM(CausalLM):
    """An end-to-end Phi-4-mini model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Phi-4-mini model, simply by calling ``fit()``.

    This model has a ``generate()`` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    ``sampler`` argument on ``compile()``. You can recompile the model with
    different ``keras_hub.samplers`` objects to control the generation. By
    default, ``"top_k"`` sampling will be used.

    Args:
        backbone: A ``keras_hub.models.Phi4MiniBackbone`` instance.
        preprocessor: A ``keras_hub.models.Phi4MiniCausalLMPreprocessor``
            or ``None``.  If ``None``, this model will not apply
            preprocessing, and inputs should be preprocessed before calling
            the model.
    """

    backbone_cls = Phi4MiniBackbone
    preprocessor_cls = Phi4MiniCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        # Use backbone.input (the full input dict), not backbone.inputs
        # (the flattened list).
        inputs = backbone.input
        hidden_states = backbone(inputs)
        # Tied-embedding output projection: reuse the embedding matrix
        # (transpose) via ReversibleEmbedding with reverse=True.
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
    ):
        """Forward pass of ``Phi4MiniCausalLM`` with KV-cache.

        ``call_with_cache`` enables efficient autoregressive inference by
        caching intermediate key/value tensors so previously seen tokens do
        not need to be reprocessed.

        Args:
            token_ids: Dense int Tensor with shape ``(batch_size, 1)`` during
                incremental decoding (or ``(batch_size, seq_len)`` for the
                seed pass).
            cache: Float Tensor of shape
                ``(batch_size, num_layers, 2, cache_len, num_kv_heads, head_dim)``.
            cache_update_index: int or int Tensor.  The position in the
                cache to write the new KV values.

        Returns:
            A ``(logits, hidden_states, cache)`` tuple.
        """
        x = self.backbone.token_embedding(token_ids)
        updated_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                attention_cache=current_cache,
                attention_cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids):
        """Build and seed an empty KV-cache."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_key_value_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_query_heads
        shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_key_value_heads,
            head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache with the prompt tokens.
        _, hidden_states, cache = self.call_with_cache(token_ids, cache, 0)
        return hidden_states, cache

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        """A compilable generation function for a single batch of inputs.

        Args:
            inputs: A dictionary with ``"token_ids"`` and ``"padding_mask"``.
            stop_token_ids: Tuple of stop-token IDs.  When all sequences have
                produced a stop token, generation halts.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        hidden_states, cache = self._build_cache(token_ids)
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index,
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        # Build an output padding mask.
        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    def generate(self, inputs, max_length=None, stop_token_ids="auto"):
        if self.preprocessor and stop_token_ids == "auto":
            # Stop at <|endoftext|> (id 199999) and <|end|> (id 200020).
            stop_token_ids = [self.preprocessor.tokenizer.end_token_id]
            end_of_turn_id = self.preprocessor.tokenizer.token_to_id("<|end|>")
            if end_of_turn_id != 0:
                stop_token_ids.append(end_of_turn_id)
        return super().generate(inputs, max_length, stop_token_ids)
