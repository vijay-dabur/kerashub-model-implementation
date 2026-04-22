import math

from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding


class Phi4MiniRotaryEmbedding(RotaryEmbedding):
    """SuScaled rotary positional encoding layer for Phi-4-mini.

    Phi-4-mini uses Long-RoPE ("Su Scaling") to extend context from the
    original pretraining length (4 096 tokens) to 128 K tokens.  Two sets
    of per-frequency rescaling factors are provided:
    ``inverese_freq_short_factor`` (used when sequence length <=
    ``pretraining_sequence_length``) and ``inverese_freq_long_factor``
    (used when sequence length > ``pretraining_sequence_length``).

    The amplitude-correction term ``embedding_scaling_factor`` follows the
    formula from the original LongRoPE paper:
    ``sqrt(1 + log(scale) / log(pretraining_sequence_length))``.

    This layer operates on the *rotary portion* of the query/key heads
    (i.e. the first ``rotary_dim`` features); the calling attention layer
    is responsible for the split-apply-concat pattern.

    Args:
        inverese_freq_short_factor (list[float]): Per-frequency divisors for
            short sequences.  Length must equal ``rotary_dim // 2``.
        inverese_freq_long_factor (list[float]): Per-frequency divisors for
            long sequences.  Length must equal ``rotary_dim // 2``.
        max_sequence_length (int): Maximum sequence length the model will be
            used with.  Used to decide which factor set to apply.
        pretraining_sequence_length (int): Context length used during
            pretraining (``original_max_position_embeddings``).
        max_wavelength (int): Base wavelength θ for standard RoPE.

    References:
        - [LongRoPE](https://arxiv.org/abs/2402.13753)
        - [Phi-4-mini HuggingFace implementation](https://huggingface.co/microsoft/phi-4-mini-instruct)
    """

    def __init__(
        self,
        inverese_freq_short_factor,
        inverese_freq_long_factor,
        max_sequence_length=131072,
        pretraining_sequence_length=4096,
        max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(max_wavelength=max_wavelength, **kwargs)
        self.max_sequence_length = max_sequence_length
        self.pretraining_sequence_length = pretraining_sequence_length

        scaling_factor = max_sequence_length / pretraining_sequence_length
        if scaling_factor <= 1.0:
            self.embedding_scaling_factor = 1.0
        else:
            self.embedding_scaling_factor = math.sqrt(
                1
                + math.log(scaling_factor)
                / math.log(self.pretraining_sequence_length)
            )

        self.inverese_freq_short_factor = inverese_freq_short_factor
        self.inverese_freq_long_factor = inverese_freq_long_factor

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        feature_axis = len(inputs.shape) - 1
        sequence_axis = 1

        rotary_dim = ops.shape(inputs)[feature_axis]
        inverse_freq = self._get_inverse_freq(rotary_dim)

        # Select the factor set based on current sequence length.
        if ops.shape(inputs)[sequence_axis] > self.pretraining_sequence_length:
            inverse_freq = ops.divide(
                inverse_freq,
                ops.convert_to_tensor(self.inverese_freq_long_factor),
            )
        else:
            inverse_freq = ops.divide(
                inverse_freq,
                ops.convert_to_tensor(self.inverese_freq_short_factor),
            )

        if positions is None:
            positions = self._compute_positions(inputs, start_index)
        else:
            positions = ops.cast(positions, "float32")

        freq = ops.einsum("i,j->ij", positions, inverse_freq)
        embedding = ops.stack((freq, freq), axis=-2)
        embedding = ops.reshape(
            embedding, (*ops.shape(freq)[:-1], ops.shape(freq)[-1] * 2)
        )

        # Reshape the embedding to be broadcastable with input shape.
        if feature_axis < sequence_axis:
            embedding = ops.transpose(embedding)
        for axis in range(len(inputs.shape)):
            if axis != sequence_axis and axis != feature_axis:
                embedding = ops.expand_dims(embedding, axis)

        cos_emb = ops.cast(
            ops.cos(embedding) * self.embedding_scaling_factor,
            self.compute_dtype,
        )
        sin_emb = ops.cast(
            ops.sin(embedding) * self.embedding_scaling_factor,
            self.compute_dtype,
        )
        return cos_emb, sin_emb

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_sequence_length": self.max_sequence_length,
                "pretraining_sequence_length": self.pretraining_sequence_length,
                "inverese_freq_short_factor": self.inverese_freq_short_factor,
                "inverese_freq_long_factor": self.inverese_freq_long_factor,
            }
        )
        return config
