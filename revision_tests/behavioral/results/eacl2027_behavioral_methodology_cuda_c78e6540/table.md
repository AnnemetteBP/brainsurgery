# Behavioral methodology correction

## Structural losslessness

This is the primary losslessness evidence. It uses only a namespace move and exact name restoration; no arithmetic or dtype conversion is performed.

| **Model** | **Dtype** | **Prompts** | **Exact tensors ↑** | **PPL ratio ↓** | **Final cosine ↑** | **Sequence cosine ↑** | **Max abs diff ↓** | **Exact generations ↑** | **MCQ agreement ↑** | **NaN** | **Inf** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia 70M | float16 | 70 | 94/94 | 1.00000000 | 0.99998998 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Pythia 410M | float16 | 70 | 364/364 | 1.00000000 | 0.99999699 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Pythia 2.8B | float16 | 70 | 484/484 | 1.00000000 | 0.99999698 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Pythia 12B | float16 | 70 | 544/544 | 1.00000000 | 0.99999690 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| GPT-2 124M | float32 | 70 | 160/160 | 1.00000000 | 0.99999984 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| GPT-2 XL 1.5B | float32 | 70 | 628/628 | 1.00000000 | 0.99999986 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| OLMo 1B (sharded) | float32 | 70 | 114/114 | 1.00000000 | 0.99999992 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| OLMo 7B (sharded, FP32) | float32 | 70 | 226/226 | 1.00000000 | 0.99999990 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Qwen2.5 0.5B | bfloat16 | 70 | 290/290 | 1.00000000 | 0.99999174 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Qwen2.5 7B (sharded, BF16) | bfloat16 | 70 | 339/339 | 1.00000000 | 0.99999312 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |

## Pythia precision ablation

This is a dtype/precision ablation, not the primary losslessness test. The old native-FP16 arithmetic round trip is not used as primary losslessness evidence.

| **Model** | **Storage/inference dtype** | **Prompts** | **Exact tensors ↑** | **PPL ratio ↓** | **Final cosine ↑** | **Sequence cosine ↑** | **Max abs diff ↓** | **Exact generations ↑** | **MCQ agreement ↑** | **NaN** | **Inf** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia 70M (native_fp16) | float16/float16 | 70 | 68/94 | 1.01559670 | 0.99998982 | 0.99999995 | 12.5 | 34/70 | 60/60 | 0 | 0 |
| Pythia 70M (fp32_throughout) | float32/float32 | 70 | 94/94 | 1.00000000 | 0.99999982 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Pythia 410M (native_fp16) | float16/float16 | 70 | 266/364 | 1.00173069 | 0.99963612 | 0.99980833 | 2.3691406 | 59/70 | 58/60 | 0 | 0 |
| Pythia 410M (fp32_throughout) | float32/float32 | 70 | 364/364 | 1.00000000 | 0.99999991 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Pythia 2.8B (native_fp16) | float16/float16 | 70 | 354/484 | 1.00102777 | 0.99989628 | 0.99989772 | 1.8027344 | 64/70 | 58/60 | 0 | 0 |
| Pythia 2.8B (fp32_throughout) | float32/float32 | 70 | 484/484 | 1.00000000 | 0.99999995 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
| Pythia 12B (native_fp16) | float16/float16 | 70 | 398/544 | 1.00223011 | 0.99962275 | 0.99959205 | 2.390625 | 49/70 | 57/60 | 0 | 0 |
| Pythia 12B (fp32_throughout) | float32/float32 | 70 | 544/544 | 1.00000000 | 0.99999992 | 1.00000000 | 0 | 70/70 | 60/60 | 0 | 0 |
