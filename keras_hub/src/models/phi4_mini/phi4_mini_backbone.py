import keras
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.phi4_mini.phi4_mini_decoder import Phi4MiniDecoder
from keras_hub.src.models.phi4_mini.phi4_mini_layernorm import Phi4MiniLayerNorm


def _phi4_mini_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.Phi4MiniBackbone")
class Phi4MiniBackbone(Backbone):
    """Phi-4-mini core network with hyperparameters.

    This network implements a Transformer-based decoder model, Phi-4-mini,
    as described in the
    ["Phi-4 Technical Report"](https://aka.ms/phi-4-multimodal/techreport).
    It includes the token embedding lookup and the stack of transformer
    decoder layers.

    The default constructor gives a fully customizable, randomly initialized
    Phi-4-mini model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the
    ``from_preset`` constructor.

    Phi-4-mini introduces two key changes over Phi-3:

    * **Grouped Query Attention (GQA)**: 24 query heads share 8 KV heads,
      reducing the KV cache size by 3×.
    * **Partial Rotary Position Embedding**: Only
      ``floor(head_dim * partial_rotary_factor)`` = 96 of the 128
      head-dimensions receive rotary encoding; the rest pass through
      unchanged.

    Args:
        vocabulary_size (int): The size of the token vocabulary.
            Phi-4-mini uses 200 064.
        num_layers (int): The number of transformer decoder blocks.
        hidden_dim (int): The size of token embeddings and transformer
            hidden states.  Production value: 3072.
        intermediate_dim (int): The MLP hidden dimension
            (``intermediate_size`` in config).  Production value: 8192.
        num_query_heads (int): The number of query attention heads.
            Production value: 24.
        num_key_value_heads (int): The number of key-value attention heads.
            Production value: 8 (GQA ratio 3:1).
        partial_rotary_factor (float, optional): Fraction of each head's
            dimension that receives RoPE.  Defaults to ``0.75`` (96/128).
        layer_norm_epsilon (float, optional): Epsilon for RMSNorm layers.
            Defaults to ``1e-5``.
        dropout (float, optional): Dropout probability.  Defaults to ``0.0``.
        max_sequence_length (int, optional): Maximum sequence length
            the model will be used with.  Defaults to ``131072`` (128 K).
        pretraining_sequence_length (int, optional): The sequence length
            used during pretraining before context extension
            (``original_max_position_embeddings`` in config).
            Defaults to ``4096``.
        rope_max_wavelength (int, optional): Base wavelength θ for RoPE.
            Defaults to ``10000``.
        rope_scaling_type (str | None, optional): ``None`` for standard RoPE,
            ``"su"`` for SuScaled / LongRoPE.  Phi-4-mini always uses
            ``"su"``.  Defaults to ``None``.
        rope_scaling_short_factor (list[float] | None, optional): Per-freq
            divisors for sequences ≤ ``pretraining_sequence_length``.
            Required when ``rope_scaling_type="su"``.
        rope_scaling_long_factor (list[float] | None, optional): Per-freq
            divisors for sequences > ``pretraining_sequence_length``.
            Required when ``rope_scaling_type="su"``.
        dtype: string or ``keras.mixed_precision.DTypePolicy``.

    Examples:

    ```python
    import numpy as np
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Phi-4-mini decoder.
    model = keras_hub.models.Phi4MiniBackbone.from_preset(
        "phi_4_mini_instruct"
    )
    model(input_data)

    # Randomly initialized Phi-4-mini decoder with custom config.
    model = keras_hub.models.Phi4MiniBackbone(
        vocabulary_size=200064,
        num_layers=32,
        hidden_dim=3072,
        intermediate_dim=8192,
        num_query_heads=24,
        num_key_value_heads=8,
        partial_rotary_factor=0.75,
        dtype="float32",
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        partial_rotary_factor=0.75,
        layer_norm_epsilon=1e-5,
        dropout=0.0,
        max_sequence_length=131072,
        pretraining_sequence_length=4096,
        rope_max_wavelength=10000,
        rope_scaling_type=None,
        rope_scaling_short_factor=None,
        rope_scaling_long_factor=None,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,  # phi-4-mini: tie_word_embeddings=True
            embeddings_initializer=_phi4_mini_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = Phi4MiniDecoder(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                partial_rotary_factor=partial_rotary_factor,
                rope_max_wavelength=rope_max_wavelength,
                layer_norm_epsilon=layer_norm_epsilon,
                activation="silu",
                kernel_initializer=_phi4_mini_kernel_initializer(stddev=0.02),
                dropout=dropout,
                max_sequence_length=max_sequence_length,
                pretraining_sequence_length=pretraining_sequence_length,
                rope_scaling_type=rope_scaling_type,
                rope_scaling_short_factor=rope_scaling_short_factor,
                rope_scaling_long_factor=rope_scaling_long_factor,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = Phi4MiniLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.pretraining_sequence_length = pretraining_sequence_length
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_short_factor = rope_scaling_short_factor
        self.rope_scaling_long_factor = rope_scaling_long_factor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_key_value_heads": self.num_key_value_heads,
                "partial_rotary_factor": self.partial_rotary_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
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
