"""Phi-4-mini model preset configurations."""

# Long-context RoPE scaling factors extracted from
# microsoft/phi-4-mini-instruct config.json.
# Both arrays have length == rotary_dim // 2 == 48
# (head_dim=128, partial_rotary_factor=0.75 → rotary_dim=96 → 48 pairs).
_PHI4_MINI_LONG_FACTOR = [
    1.0, 1.118320672, 1.250641126, 1.398617824, 1.564103225, 1.74916897,
    1.956131817, 2.187582649, 2.446418898, 2.735880826, 3.059592084,
    3.421605075, 3.826451687, 4.279200023, 4.785517845, 5.351743533,
    5.984965424, 6.693110555, 7.485043894, 8.370679318, 9.36110372,
    10.4687158, 11.70738129, 13.09260651, 14.64173252, 16.37415215,
    18.31155283, 20.47818807, 22.90118105, 25.61086418, 28.64115884,
    32.03, 32.1, 32.13, 32.23, 32.6, 32.61, 32.64, 32.66, 32.7, 32.71,
    32.93, 32.97, 33.28, 33.49, 33.5, 44.16, 47.77,
]

_PHI4_MINI_SHORT_FACTOR = [1.0] * 48  # Standard RoPE for short contexts.

backbone_presets = {
    "phi_4_mini_instruct": {
        "metadata": {
            "description": (
                "3.8 billion parameters, 32 layers, 128k context length, "
                "Phi-4-mini instruct model by Microsoft. Uses Grouped Query "
                "Attention (24 query / 8 KV heads) and partial RoPE with "
                "LongRoPE scaling.  Vocabulary size: 200 064 (o200k_base BPE)."
            ),
            "params": 3800000000,
            "path": "phi4_mini",
        },
        "kaggle_handle": "kaggle://keras/phi4_mini/keras/phi_4_mini_instruct/1",
    },
}
