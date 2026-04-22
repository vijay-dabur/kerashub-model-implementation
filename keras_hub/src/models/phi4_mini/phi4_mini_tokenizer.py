from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Phi4MiniTokenizer",
        "keras_hub.models.Phi4MiniTokenizer",
    ]
)
class Phi4MiniTokenizer(BytePairTokenizer):
    """Phi-4-mini tokenizer layer based on Byte-Pair Encoding.

    This tokenizer class will tokenize raw strings into integer sequences
    and is based on ``keras_hub.tokenizers.BytePairTokenizer``. It uses the
    ``o200k_base`` tiktoken vocabulary (same as GPT-4o) extended with
    Phi-4-mini–specific special tokens, giving a vocabulary size of
    **200 064**.

    Unlike the underlying tokenizer, it checks for all special tokens
    needed by Phi-4-mini models and provides a ``from_preset()`` method to
    automatically download a matching vocabulary for a Phi-4-mini preset.

    If input is a batch of strings (rank > 0), the layer will output a
    ``tf.RaggedTensor`` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    ``tf.Tensor`` with static shape ``[None]``.

    Args:
        vocabulary: Either a ``string`` path to a ``vocab.json`` file, or a
            dict mapping tokens to their integer IDs.
        merges: Either a ``string`` path to a ``merges.txt`` file, or a
            list of BPE merge pairs.

    Special tokens
    --------------
    ==================== ======== ==========================================
    Token                ID       Purpose
    ==================== ======== ==========================================
    ``<|endoftext|>``    199999   BOS / EOS / PAD
    ``<|end|>``          200020   End-of-turn marker
    ``<|user|>``         200011   User turn header
    ``<|assistant|>``    200001   Assistant turn header
    ``<|system|>``       200006   System prompt header
    ``<|tool|>``         200007   Tool definition open tag
    ``<|/tool|>``        200008   Tool definition close tag
    ``<|tool_call|>``    200009   Tool call result marker
    ==================== ======== ==========================================

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.Phi4MiniTokenizer.from_preset(
        "phi_4_mini_instruct",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = Phi4MiniBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # Register special tokens before calling super().__init__ so they
        # are available during vocabulary loading.
        self._add_special_token("<|endoftext|>", "start_token")
        self._add_special_token("<|endoftext|>", "end_token")
        # Additional Phi-4-mini chat special tokens.
        self._add_special_token("<|end|>", "end_of_turn_token")
        self._add_special_token("<|user|>", "user_token")
        self._add_special_token("<|assistant|>", "assistant_token")
        self._add_special_token("<|system|>", "system_token")
        self.pad_token_id = 199999
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
