"""Load out/T2/model.safetensors as OLMo with 15 heads and run a forward pass."""
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
cfg = AutoConfig.from_pretrained(ROOT / "inputs" / "base")
cfg.num_attention_heads = 15
cfg.num_key_value_heads = 15
cfg.head_dim = 128
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
sd = load_file(str(ROOT / "out" / "T2" / "model.safetensors"))
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [m for m in missing if m != "lm_head.weight"]  # tied embeddings if any
print("missing:", missing, "unexpected:", unexpected)
assert not unexpected
tok = AutoTokenizer.from_pretrained(ROOT / "inputs" / "base")
ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits
print("logits", tuple(logits.shape), "finite:", bool(torch.isfinite(logits).all()))
print("next token:", repr(tok.decode(logits[0, -1].argmax())))
