from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone
from keras_hub.src.models.phi4_mini.phi4_mini_tokenizer import Phi4MiniTokenizer


@keras_hub_export("keras_hub.models.Phi4MiniCausalLMPreprocessor")
class Phi4MiniCausalLMPreprocessor(CausalLMPreprocessor):
    """Phi-4-mini Causal LM preprocessor.

    This preprocessing layer is meant for use with
    ``keras_hub.models.Phi4MiniCausalLM``. By default, it will take in
    batches of strings, and return outputs in a ``(x, y, sample_weight)``
    format, where the ``y`` label is the next token id in the ``x``
    sequence.

    For use with generation, the layer also exposes two methods
    ``generate_preprocess()`` and ``generate_postprocess()``. When this
    preprocessor is attached to a ``keras_hub.models.Phi4MiniCausalLM``
    instance, these methods will be called implicitly in ``generate()``.
    They can also be called standalone (e.g. to precompute preprocessing
    inputs for generation in a separate process).

    Args:
        tokenizer: A ``keras_hub.models.Phi4MiniTokenizer`` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If ``True``, the preprocessor will prepend the
            tokenizer start token to each input sequence. Default is
            ``True``.
        add_end_token: If ``True``, the preprocessor will append the
            tokenizer end token to each input sequence. Default is
            ``False``.

    Call arguments:
        x: A string, ``tf.Tensor`` or list of python strings.
        y: Label data. Should always be ``None`` as the layer generates
            labels.
        sample_weight: Label weights. Should always be ``None`` as the
            layer generates label weights.
        sequence_length: Pass to override the configured
            ``sequence_length`` of the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.Phi4MiniCausalLMPreprocessor.from_preset(
        "phi_4_mini_instruct"
    )

    # Tokenize and pack a single sentence.
    preprocessor("Explain quantum entanglement.")

    # Tokenize a batch of sentences.
    preprocessor(["Hello world", "What is AI?"])

    # Map a dataset.
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices(["text1", "text2"])
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = Phi4MiniBackbone
    tokenizer_cls = Phi4MiniTokenizer
