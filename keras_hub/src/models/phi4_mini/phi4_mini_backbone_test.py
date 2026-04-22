import pytest
from keras import ops

from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone
from keras_hub.src.tests.test_case import TestCase


class Phi4MiniBackboneTest(TestCase):
    def setUp(self):
        # Tiny config — fast on CPU, exercises the full pipeline.
        # hidden_dim=32, num_query_heads=4 → head_dim=8
        # partial_rotary_factor=0.75 → rotary_dim=6, num_freq_pairs=3
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 32,
            "intermediate_dim": 64,
        }
        # Config with SuScaled RoPE (LongRoPE) enabled.
        self.su_rotary_init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 32,
            "intermediate_dim": 64,
            "partial_rotary_factor": 0.75,
            "max_sequence_length": 20,
            "pretraining_sequence_length": 10,
            "rope_scaling_type": "su",
            # rotary_dim = floor(8 * 0.75) = 6  →  num_freq_pairs = 3
            "rope_scaling_short_factor": [1.0, 1.2, 1.5],
            "rope_scaling_long_factor": [1.0, 2.0, 3.0],
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Phi4MiniBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 32),
        )

    def test_backbone_basics_with_su_rope(self):
        self.run_backbone_test(
            cls=Phi4MiniBackbone,
            init_kwargs=self.su_rotary_init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 32),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Phi4MiniBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_saved_model_with_su_rope(self):
        self.run_model_saving_test(
            cls=Phi4MiniBackbone,
            init_kwargs=self.su_rotary_init_kwargs,
            input_data=self.input_data,
        )
