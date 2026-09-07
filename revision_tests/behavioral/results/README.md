# Behavioral result records

The `linux_99693f2/` result is an auxiliary multiply-by-one serialization check.
It is **not** the expanded version of the original paper's behavioral analysis
and must not be used to resolve the behavioral reviewer concern.

The reportable behavioral result must be produced by
`run_cuda_paper_matrix.py`. That evaluator retains the original PPL,
final-token cosine, full-sequence cosine/difference, top-1, and generation
metrics while expanding coverage to the sourced 70-prompt, ten-model matrix.
It creates no paper table or paper prose unless every required per-prompt and
aggregate field is present and every frozen threshold passes.

The completed v4 native-dtype matrix is preserved under
`eacl2027_behavioral_paper_cuda_4cb30d71/`. It is non-reportable under the
frozen thresholds because the four FP16 Pythia round trips failed at least one
behavioral-preservation threshold; the FP32 GPT-2/OLMo and BF16 Qwen cases
passed. No paper table or result prose was generated.
