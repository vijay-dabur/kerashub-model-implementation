"""Convert microsoft/phi-4-mini-instruct HuggingFace weights to KerasHub.

Usage
-----
    python tools/checkpoint_conversion/convert_phi4_mini_checkpoints.py \\
        --preset phi_4_mini_instruct \\
        --upload_uri kaggle://keras/phi4_mini/keras/phi_4_mini_instruct

Prerequisites
-------------
    pip install transformers torch huggingface_hub

Weight mapping (HuggingFace → Keras)
--------------------------------------
HuggingFace (Phi3ForCausalLM)                          Keras variable path
──────────────────────────────────────────────────────────────────────────
model.embed_tokens.weight                        token_embedding/embeddings
model.layers.{i}.self_attn.q_proj.weight  transformer_layer_{i}/attention/query/kernel
model.layers.{i}.self_attn.k_proj.weight  transformer_layer_{i}/attention/key/kernel
model.layers.{i}.self_attn.v_proj.weight  transformer_layer_{i}/attention/value/kernel
model.layers.{i}.self_attn.o_proj.weight  transformer_layer_{i}/attention/attention_output/kernel
model.layers.{i}.mlp.gate_proj.weight     transformer_layer_{i}/feedforward_gate_dense/kernel
model.layers.{i}.mlp.up_proj.weight       transformer_layer_{i}/feedforward_intermediate_dense/kernel
model.layers.{i}.mlp.down_proj.weight     transformer_layer_{i}/feedforward_output_dense/kernel
model.layers.{i}.input_layernorm.weight   transformer_layer_{i}/pre_attention_layernorm/scale
model.layers.{i}.post_attention_layernorm.weight transformer_layer_{i}/post_attention_layernorm/scale
model.norm.weight                          sequence_output_layernorm/scale
lm_head.weight                             (skipped — tied to token_embedding)

Note on weight shapes
---------------------
HuggingFace stores projection kernels as [out_features, in_features].
Keras Dense/EinsumDense kernels are [in_features, out_features] for
standard Dense layers, but EinsumDense with equation ``"bqm,muh->bquh"``
uses kernel shape ``[hidden_dim, num_heads, head_dim]`` (i.e. the "m"
dimension first).  Therefore all projection matrices are reshaped and/or
transposed during conversion.
"""

import argparse
import gc
import os

import numpy as np

os.environ["KERAS_BACKEND"] = "torch"

import huggingface_hub  # noqa: E402
import keras  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from keras_hub import upload_preset  # noqa: E402
from keras_hub.src.models.phi4_mini.phi4_mini_backbone import (  # noqa: E402
    Phi4MiniBackbone,
)
from keras_hub.src.models.phi4_mini.phi4_mini_causal_lm import (  # noqa: E402
    Phi4MiniCausalLM,
)
from keras_hub.src.models.phi4_mini.phi4_mini_causal_lm_preprocessor import (  # noqa: E402
    Phi4MiniCausalLMPreprocessor,
)
from keras_hub.src.models.phi4_mini.phi4_mini_presets import (  # noqa: E402
    _PHI4_MINI_LONG_FACTOR,
    _PHI4_MINI_SHORT_FACTOR,
)
from keras_hub.src.models.phi4_mini.phi4_mini_tokenizer import (  # noqa: E402
    Phi4MiniTokenizer,
)

PRESET_MAP = {
    "phi_4_mini_instruct": "microsoft/phi-4-mini-instruct",
}

# Production backbone config (matches HuggingFace config.json exactly).
BACKBONE_CONFIG = {
    "vocabulary_size": 200064,
    "num_layers": 32,
    "num_query_heads": 24,
    "num_key_value_heads": 8,
    "hidden_dim": 3072,
    "intermediate_dim": 8192,
    "partial_rotary_factor": 0.75,
    "layer_norm_epsilon": 1e-5,
    "dropout": 0.0,
    "max_sequence_length": 131072,
    "pretraining_sequence_length": 4096,
    "rope_max_wavelength": 10000,
    "rope_scaling_type": "su",
    "rope_scaling_short_factor": _PHI4_MINI_SHORT_FACTOR,
    "rope_scaling_long_factor": _PHI4_MINI_LONG_FACTOR,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Phi-4-mini HuggingFace checkpoint to KerasHub."
    )
    p.add_argument(
        "--preset",
        default="phi_4_mini_instruct",
        choices=list(PRESET_MAP.keys()),
        help="Preset name to convert.",
    )
    p.add_argument(
        "--upload_uri",
        default=None,
        help=(
            "Kaggle URI to upload the converted preset to, e.g. "
            "``kaggle://keras/phi4_mini/keras/phi_4_mini_instruct``."
        ),
    )
    return p.parse_args()


def download_hf_model(hf_model_name, extract_dir):
    hf_model_dir = huggingface_hub.snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=["*.json", "*.safetensors", "*.model", "merges.txt"],
        ignore_patterns=["*/*"],
        local_dir=extract_dir,
    )
    return hf_model_dir


def convert_tokenizer(hf_model_dir):
    """Build a Phi4MiniTokenizer from the HuggingFace tokenizer files."""
    vocab_path = os.path.join(hf_model_dir, "vocab.json")
    merges_path = os.path.join(hf_model_dir, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(
            "Could not find vocab.json or merges.txt in "
            f"{hf_model_dir}.  Phi-4-mini uses a tiktoken-based BPE "
            "tokenizer; ensure the HuggingFace snapshot contains these "
            "files (they may need to be extracted from the tokenizer.json "
            "using transformers' tokenizer.save_pretrained())."
        )
    return Phi4MiniTokenizer(vocabulary=vocab_path, merges=merges_path)


def convert_backbone(hf_model_dir):
    """Build a Phi4MiniBackbone and load weights from HuggingFace."""
    print("Building Keras Phi4MiniBackbone …")
    keras_model = Phi4MiniBackbone(**BACKBONE_CONFIG)

    print("Loading HuggingFace model …")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.eval()
    hf_state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}
    del hf_model
    gc.collect()

    print("Mapping weights …")
    # Token embedding: shape [vocab_size, hidden_dim] — no transpose needed.
    keras_model.token_embedding.embeddings.assign(
        hf_state["model.embed_tokens.weight"]
    )

    # Final RMSNorm scale: shape [hidden_dim].
    keras_model.layer_norm.scale.assign(hf_state["model.norm.weight"])

    head_dim = BACKBONE_CONFIG["hidden_dim"] // BACKBONE_CONFIG["num_query_heads"]
    num_q = BACKBONE_CONFIG["num_query_heads"]
    num_kv = BACKBONE_CONFIG["num_key_value_heads"]

    for i, layer in enumerate(keras_model.transformer_layers):
        attn = layer.attention

        # Q projection: HF [num_q * head_dim, hidden] → Keras [hidden, num_q, head_dim]
        q_w = hf_state[f"model.layers.{i}.self_attn.q_proj.weight"]
        q_w = q_w.reshape(num_q, head_dim, BACKBONE_CONFIG["hidden_dim"])
        q_w = q_w.transpose(2, 0, 1)  # [hidden, num_q, head_dim]
        attn.query_dense.kernel.assign(q_w)

        # K projection: HF [num_kv * head_dim, hidden] → Keras [hidden, num_kv, head_dim]
        k_w = hf_state[f"model.layers.{i}.self_attn.k_proj.weight"]
        k_w = k_w.reshape(num_kv, head_dim, BACKBONE_CONFIG["hidden_dim"])
        k_w = k_w.transpose(2, 0, 1)  # [hidden, num_kv, head_dim]
        attn.key_dense.kernel.assign(k_w)

        # V projection: same shape as K.
        v_w = hf_state[f"model.layers.{i}.self_attn.v_proj.weight"]
        v_w = v_w.reshape(num_kv, head_dim, BACKBONE_CONFIG["hidden_dim"])
        v_w = v_w.transpose(2, 0, 1)
        attn.value_dense.kernel.assign(v_w)

        # O projection: HF [hidden, num_q * head_dim] → Keras [num_q, head_dim, hidden]
        o_w = hf_state[f"model.layers.{i}.self_attn.o_proj.weight"]
        o_w = o_w.reshape(BACKBONE_CONFIG["hidden_dim"], num_q, head_dim)
        o_w = o_w.transpose(1, 2, 0)  # [num_q, head_dim, hidden]
        attn.output_dense.kernel.assign(o_w)

        # MLP: HF Dense kernels are [out, in] — transpose to [in, out].
        layer.feedforward_gate_dense.kernel.assign(
            hf_state[f"model.layers.{i}.mlp.gate_proj.weight"].T
        )
        layer.feedforward_intermediate_dense.kernel.assign(
            hf_state[f"model.layers.{i}.mlp.up_proj.weight"].T
        )
        layer.feedforward_output_dense.kernel.assign(
            hf_state[f"model.layers.{i}.mlp.down_proj.weight"].T
        )

        # Layer norms: shape [hidden_dim] — no transform needed.
        layer.pre_attention_layernorm.scale.assign(
            hf_state[f"model.layers.{i}.input_layernorm.weight"]
        )
        layer.post_attention_layernorm.scale.assign(
            hf_state[f"model.layers.{i}.post_attention_layernorm.weight"]
        )

    print("Weights mapped successfully.")
    return keras_model


def validate(keras_model, hf_model_dir, num_tokens=16, tolerance=1e-2):
    """Numerical comparison between Keras and HuggingFace logits."""
    print(f"Running validation (tolerance={tolerance}) …")
    causal_lm = Phi4MiniCausalLM(backbone=keras_model)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).eval()

    token_ids_np = np.arange(1, num_tokens + 1, dtype=np.int32)[None, :]
    token_ids_pt = torch.from_numpy(token_ids_np).long()

    with torch.no_grad():
        hf_logits = hf_model(token_ids_pt).logits.numpy()

    keras_inputs = {
        "token_ids":    token_ids_np,
        "padding_mask": np.ones_like(token_ids_np),
    }
    from keras import ops

    keras_logits = ops.convert_to_numpy(causal_lm(keras_inputs))

    abs_diff = np.abs(hf_logits - keras_logits)
    max_diff  = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    print(f"  max  |HF − Keras| = {max_diff:.6f}")
    print(f"  mean |HF − Keras| = {mean_diff:.6f}")
    if max_diff > tolerance:
        raise RuntimeError(
            f"Validation FAILED: max diff {max_diff:.6f} > tol {tolerance}."
        )
    print("  Validation PASSED ✓")


def main():
    args = parse_args()
    hf_model_name = PRESET_MAP[args.preset]
    extract_dir = os.path.join("./hf_downloads", args.preset)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Downloading {hf_model_name} …")
    hf_model_dir = download_hf_model(hf_model_name, extract_dir)

    tokenizer = convert_tokenizer(hf_model_dir)
    backbone  = convert_backbone(hf_model_dir)
    validate(backbone, hf_model_dir)

    preprocessor = Phi4MiniCausalLMPreprocessor(tokenizer)
    causal_lm     = Phi4MiniCausalLM(backbone=backbone, preprocessor=preprocessor)

    if args.upload_uri:
        print(f"Uploading preset to {args.upload_uri} …")
        upload_preset(args.upload_uri, causal_lm)
        print("Upload complete.")
    else:
        save_path = os.path.join("./converted_presets", args.preset)
        os.makedirs(save_path, exist_ok=True)
        causal_lm.save_to_preset(save_path)
        print(f"Saved preset to: {save_path}")


if __name__ == "__main__":
    main()
