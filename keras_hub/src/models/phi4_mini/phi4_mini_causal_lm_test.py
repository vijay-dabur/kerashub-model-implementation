import os
from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone
from keras_hub.src.models.phi4_mini.phi4_mini_causal_lm import Phi4MiniCausalLM
from keras_hub.src.models.phi4_mini.phi4_mini_causal_lm_preprocessor import (
    Phi4MiniCausalLMPreprocessor,
)
from keras_hub.src.models.phi4_mini.phi4_mini_tokenizer import Phi4MiniTokenizer
from keras_hub.src.tests.test_case import TestCase


class Phi4MiniCausalLMTest(TestCase):
    def setUp(self):
        # NOTE: Phi4MiniTokenizer uses BytePairTokenizer (BPE).
        # The test relies on a tiny BPE vocabulary file located in the
        # test data directory.  Create ``phi4_mini_test_vocab.json`` and
        # ``phi4_mini_test_merges.txt`` alongside this test file, or point
        # to an existing GPT-2-style vocabulary for testing purposes.
        #
        # For CI/CD without real vocabulary files, a lightweight integration
        # approach is to subclass BytePairTokenizer with a hardcoded tiny
        # vocabulary.  The test below is structured to be skipped gracefully
        # when the vocab files are absent.
        vocab_path = os.path.join(
            self.get_test_data_dir(), "phi4_mini_test_vocab.json"
        )
        merges_path = os.path.join(
            self.get_test_data_dir(), "phi4_mini_test_merges.txt"
        )
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            self.skipTest(
                "Phi-4-mini test vocabulary files not found. "
                "Please provide phi4_mini_test_vocab.json and "
                "phi4_mini_test_merges.txt in the test data directory."
            )

        self.preprocessor = Phi4MiniCausalLMPreprocessor(
            Phi4MiniTokenizer(
                vocabulary=vocab_path,
                merges=merges_path,
            ),
            sequence_length=12,
        )
        self.vocab_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = Phi4MiniBackbone(
            vocabulary_size=self.vocab_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=32,
            intermediate_dim=64,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["the quick brown fox", "the earth is round"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Phi4MiniCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 12, self.vocab_size),
        )

    def test_generate(self):
        causal_lm = Phi4MiniCausalLM(**self.init_kwargs)
        # String input.
        prompt = "the fox"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            prompt_ids["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            prompt_ids["padding_mask"][:, :5],
        )

    def test_early_stopping(self):
        causal_lm = Phi4MiniCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favour end_token_id."""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the fox", "the earth"]
            output = causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = Phi4MiniCausalLM(**self.init_kwargs)
        causal_lm.generate("the fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Phi4MiniCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        model = Phi4MiniCausalLM(**self.init_kwargs)
        input_data = self.input_data.copy()
        if "padding_mask" in input_data:
            input_data["padding_mask"] = ops.cast(
                input_data["padding_mask"], "int32"
            )
        expected_output_shape = (
            2,
            12,
            self.preprocessor.tokenizer.vocabulary_size(),
        )
        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )

    def test_get_quantization_layer_structure(self):
        causal_lm = Phi4MiniCausalLM(**self.init_kwargs)
        structure = causal_lm.get_quantization_layer_structure("gptq")
        self.assertIsInstance(structure, dict)
        self.assertIn("pre_block_layers", structure)
        self.assertIn("sequential_blocks", structure)
        self.assertEqual(
            structure["pre_block_layers"],
            [self.backbone.token_embedding],
        )
        self.assertEqual(
            structure["sequential_blocks"], self.backbone.transformer_layers
        )
        self.assertIsNone(causal_lm.get_quantization_layer_structure("int8"))

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Phi4MiniCausalLM.presets:
            self.run_preset_test(
                cls=Phi4MiniCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
