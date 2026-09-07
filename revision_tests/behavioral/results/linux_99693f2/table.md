# Ten-checkpoint CUDA behavioral regression

Protocol: `eacl2027_behavioral_matrix_v2`  
Run: `eacl2027_behavioral_matrix_cuda_99693f2`  
Commit: `99693f26716da7ffbe4109b20bcdcd475c008022`  
GPU: NVIDIA B200  
Status: **REPORTABLE PAPER EVIDENCE — ALL AUTOMATED EVALUATION GATES PASSED**

| Model | Parameters | Dtype | Exact tensors | Exact logits | Top-1 | Greedy | MCQ agreement |
|---|---:|---|---:|---:|---:|---:|---:|
| Pythia 70M | 70M | float16 | 94/94 | 70/70 | 70/70 | 70/70 | 60/60 |
| Pythia 410M | 410M | float16 | 364/364 | 70/70 | 70/70 | 70/70 | 60/60 |
| Pythia 2.8B | 2.80B | float16 | 484/484 | 70/70 | 70/70 | 70/70 | 60/60 |
| Pythia 12B | 12.00B | float16 | 544/544 | 70/70 | 70/70 | 70/70 | 60/60 |
| GPT-2 124M | 124M | float32 | 160/160 | 70/70 | 70/70 | 70/70 | 60/60 |
| GPT-2 XL 1.5B | 1.50B | float32 | 628/628 | 70/70 | 70/70 | 70/70 | 60/60 |
| OLMo 1B (sharded) | 1.00B | float32 | 114/114 | 70/70 | 70/70 | 70/70 | 60/60 |
| OLMo 7B (sharded, FP32) | 7.00B | float32 | 226/226 | 70/70 | 70/70 | 70/70 | 60/60 |
| Qwen2.5 0.5B | 500M | bfloat16 | 290/290 | 70/70 | 70/70 | 70/70 | 60/60 |
| Qwen2.5 7B (sharded, BF16) | 7.00B | bfloat16 | 339/339 | 70/70 | 70/70 | 70/70 | 60/60 |

Aggregate: 3243/3243 tensors were byte-exact and all 700/700 prompt pairs had exact final-token logits.

Measured endpoints:

- **Exact tensors:** every output tensor matched the independent expected
  tensor byte-for-byte before inference.
- **Exact logits:** the complete final-prompt-position vocabulary-logit vector,
  copied to float32 on CPU, was byte-identical for the reference and
  transformed checkpoints.
- **Top-1:** the argmax token of that vector matched.
- **Greedy:** all 32 greedily generated token IDs matched.
- **MCQ agreement:** the predicted label matched for each of the 60 Belebele
  and MMLU prompts per model.

The analyzer computes final-token cosine as a secondary diagnostic, but the
matrix summary retained the stronger byte-exact result rather than an aggregate
cosine value. Byte-identical nonzero vectors have mathematical cosine 1.0, so
the exact-logit endpoint establishes the same conclusion more strictly, but a
separately aggregated measured cosine value was not preserved. Perplexity and
full-sequence per-position logit cosine were not measured in this run and must
not be reported as new results.

No human scoring of behavior is required. The ordinary final submission check
for identifying metadata and unsupported wording is separate from evaluating
these results.

This is paper evidence for lossless behavioral preservation on the enumerated
rewrite, checkpoints, prompts, software environment, and GPU. It is not a
downstream-quality evaluation of intentionally lossy transformations.
