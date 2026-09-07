# Non-reportable behavioral evidence

This run completed the full ten-model, 70-prompt matrix on NVIDIA B200 at Git
commit `4cb30d7178c8d5d70f334313b4e72799e1e0a5b1`.

All required metrics were finite, every independent tensor oracle passed, and
all imperative-equivalence comparisons passed. The four native-FP16 Pythia
forward--backward comparisons each missed at least one frozen behavioral
preservation threshold. The FP32 GPT-2/OLMo and BF16 Qwen comparisons passed.

The complete sanitized data is in `evidence.json`. This directory deliberately
contains no paper table or paste-ready result prose.
