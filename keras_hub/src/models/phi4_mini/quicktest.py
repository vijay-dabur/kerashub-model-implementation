"""Standalone smoke test for Phi-4-mini — no pytest or test fixtures needed.

Run from the keras-hub/ directory after installing dependencies:

    python keras_hub/src/models/phi4_mini/quicktest.py

Expected output (all PASSED):
    [1/6] Import ................. PASSED
    [2/6] Backbone basics ........ PASSED
    [3/6] Standard RoPE .......... PASSED
    [4/6] SuScaled (LongRoPE) .... PASSED
    [5/6] GQA validation ......... PASSED
    [6/6] CausalLM logits ........ PASSED
    All 6 tests passed.
"""

import os
import sys
import traceback

import numpy as np

# Allow running from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _check(name, fn):
    """Run fn(), print PASSED/FAILED."""
    try:
        fn()
        print(f"  {name:<28} PASSED")
        return True
    except Exception as e:
        print(f"  {name:<28} FAILED")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 1 — imports
# ---------------------------------------------------------------------------
def test_imports():
    from keras_hub.src.models.phi4_mini.phi4_mini_attention import Phi4MiniAttention  # noqa: F401
    from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone  # noqa: F401
    from keras_hub.src.models.phi4_mini.phi4_mini_causal_lm import Phi4MiniCausalLM  # noqa: F401
    from keras_hub.src.models.phi4_mini.phi4_mini_decoder import Phi4MiniDecoder  # noqa: F401
    from keras_hub.src.models.phi4_mini.phi4_mini_layernorm import Phi4MiniLayerNorm  # noqa: F401
    from keras_hub.src.models.phi4_mini.phi4_mini_presets import backbone_presets  # noqa: F401
    from keras_hub.src.models.phi4_mini.phi4_mini_rotary_embedding import Phi4MiniRotaryEmbedding  # noqa: F401
    from keras_hub.src.models.phi4_mini.phi4_mini_tokenizer import Phi4MiniTokenizer  # noqa: F401


# ---------------------------------------------------------------------------
# Shared tiny backbone config (fast on CPU)
# hidden_dim=32, num_query_heads=4 → head_dim=8
# partial_rotary_factor=0.75 → rotary_dim=6, freq_pairs=3
# ---------------------------------------------------------------------------
TINY_CFG = dict(
    vocabulary_size=50,
    num_layers=2,
    num_query_heads=4,
    num_key_value_heads=2,
    hidden_dim=32,
    intermediate_dim=64,
)


# ---------------------------------------------------------------------------
# Test 2 — backbone output shape (standard RoPE)
# ---------------------------------------------------------------------------
def test_backbone_basics():
    from keras import ops
    from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone

    model = Phi4MiniBackbone(**TINY_CFG)
    inputs = {
        "token_ids":    np.ones((2, 8), dtype=np.int32),
        "padding_mask": np.ones((2, 8), dtype=np.int32),
    }
    out = model(inputs)
    assert out.shape == (2, 8, 32), f"Expected (2,8,32), got {out.shape}"
    out_np = ops.convert_to_numpy(out)
    assert np.all(np.isfinite(out_np)), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 3 — variable sequence length
# ---------------------------------------------------------------------------
def test_variable_sequence_length():
    from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone

    model = Phi4MiniBackbone(**TINY_CFG)
    for seq_len in [1, 5, 16]:
        inputs = {
            "token_ids":    np.ones((1, seq_len), dtype=np.int32),
            "padding_mask": np.ones((1, seq_len), dtype=np.int32),
        }
        out = model(inputs)
        assert out.shape == (1, seq_len, 32), (
            f"seq_len={seq_len}: expected (1,{seq_len},32), got {out.shape}"
        )


# ---------------------------------------------------------------------------
# Test 4 — SuScaled RoPE (LongRoPE)
# ---------------------------------------------------------------------------
def test_su_scaled_rope():
    from keras import ops
    from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone

    su_cfg = dict(
        **TINY_CFG,
        partial_rotary_factor=0.75,      # rotary_dim = 6
        max_sequence_length=20,
        pretraining_sequence_length=10,
        rope_scaling_type="su",
        rope_scaling_short_factor=[1.0, 1.2, 1.5],   # len = rotary_dim//2 = 3
        rope_scaling_long_factor=[1.0, 2.0, 3.0],
    )
    model = Phi4MiniBackbone(**su_cfg)
    inputs = {
        "token_ids":    np.ones((2, 5), dtype=np.int32),
        "padding_mask": np.ones((2, 5), dtype=np.int32),
    }
    out = model(inputs)
    assert out.shape == (2, 5, 32), f"Expected (2,5,32), got {out.shape}"
    out_np = ops.convert_to_numpy(out)
    assert np.all(np.isfinite(out_np)), "SuScaled output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 5 — GQA: causal property (pos i must NOT depend on pos j > i)
# ---------------------------------------------------------------------------
def test_causal_masking():
    from keras import ops
    from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone

    model = Phi4MiniBackbone(**TINY_CFG)
    mask = np.ones((1, 8), dtype=np.int32)
    ids_a = np.ones((1, 8), dtype=np.int32)
    ids_b = ids_a.copy()
    ids_b[0, 5] = 7   # mutate a future position

    out_a = ops.convert_to_numpy(model({"token_ids": ids_a, "padding_mask": mask}))
    out_b = ops.convert_to_numpy(model({"token_ids": ids_b, "padding_mask": mask}))

    # Positions 0-4 must be identical (causal: they cannot see position 5).
    np.testing.assert_allclose(
        out_a[:, :5, :], out_b[:, :5, :], atol=1e-5,
        err_msg="Causal masking broken: early positions changed when a future token changed"
    )


# ---------------------------------------------------------------------------
# Test 6 — CausalLM logit shape and tied embeddings
# ---------------------------------------------------------------------------
def test_causal_lm():
    from keras import ops
    from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone
    from keras_hub.src.models.phi4_mini.phi4_mini_causal_lm import Phi4MiniCausalLM

    backbone = Phi4MiniBackbone(**TINY_CFG)
    lm = Phi4MiniCausalLM(backbone=backbone)

    inputs = {
        "token_ids":    np.ones((2, 6), dtype=np.int32),
        "padding_mask": np.ones((2, 6), dtype=np.int32),
    }
    logits = lm(inputs)
    assert logits.shape == (2, 6, TINY_CFG["vocabulary_size"]), (
        f"Expected (2,6,{TINY_CFG['vocabulary_size']}), got {logits.shape}"
    )
    logits_np = ops.convert_to_numpy(logits)
    assert np.all(np.isfinite(logits_np)), "Logits contain NaN or Inf"

    # Tied embeddings: LM has no extra projection weight beyond the backbone.
    n_backbone = sum(np.prod(w.shape) for w in backbone.trainable_weights)
    n_lm       = sum(np.prod(w.shape) for w in lm.trainable_weights)
    assert n_lm == n_backbone, (
        f"Tied embeddings broken: backbone has {n_backbone} params but "
        f"CausalLM has {n_lm} (expected equal)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TESTS = [
    ("[1/6] Import",            test_imports),
    ("[2/6] Backbone basics",   test_backbone_basics),
    ("[3/6] Variable seq_len",  test_variable_sequence_length),
    ("[4/6] SuScaled (LongRoPE)", test_su_scaled_rope),
    ("[5/6] Causal masking",    test_causal_masking),
    ("[6/6] CausalLM logits",   test_causal_lm),
]

if __name__ == "__main__":
    backend = os.environ.get("KERAS_BACKEND", "not set")
    print(f"\nRunning Phi-4-mini smoke tests  (KERAS_BACKEND={backend})\n")
    passed = sum(_check(name, fn) for name, fn in TESTS)
    total  = len(TESTS)
    print(f"\n{'All' if passed == total else passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)
