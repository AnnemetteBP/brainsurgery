# FlexMore Examples

This directory now contains a paper-ready OLMo-1B dense-to-expert-MoE example
with three complementary pieces:

1. `olmo_1b_0724_hf_dense_to_expert_moe.yaml`
   BrainSurgery conversion plan that rewrites the dense HF checkpoint into a
   2-expert MoE-style checkpoint by:
   - preserving shared tensors from `m0`
   - asserting `m1` matches on shared tensors
   - rewriting every layer's dense MLP weights into `experts.0` / `experts.1`
   - creating a zero router tensor `mlp.gate.weight` per layer
   - deleting the original dense MLP weights

2. `olmo_1b_0724_hf_dense_to_expert_moe_reference.py`
   Plain Python/Torch reference implementation of the same conversion semantics.
   This is useful when presenting the BrainSurgery plan against a familiar
   imperative baseline.

3. `olmo_1b_0724_hf_dense_to_expert_moe_validate.yaml`
   Validation plan that diffs the BrainSurgery output against the reference
   output.

## Suggested demo flow

Use fresh output directories when rerunning the demo. Reusing an older output
directory can make the final diff confusing if stale shards from a previous run
are still present.

Run the whole paper demo in one command:

```bash
python3 flexmore_examples/run_olmo_1b_0724_demo.py
```

Useful variants:

```bash
python3 flexmore_examples/run_olmo_1b_0724_demo.py --skip-diff
python3 flexmore_examples/run_olmo_1b_0724_demo.py --python $(which python3)
```

Or run the three stages manually:

Run the BrainSurgery conversion:

```bash
brainsurgery flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe.yaml
```

If you prefer module invocation from the repo checkout, use:

```bash
python3 -m brainsurgery.cli flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe.yaml
```

Build the reference output with matching semantics:

```bash
python3 flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe_reference.py \
  --model-a models/olmo_1b_0724_hf_dense \
  --model-b models/olmo_1b_0724_hf_dense \
  --target models/olmo_1b_0724_hf_dense_moe_reference \
  --copy-metadata \
  --write-example-config
```

Diff the two outputs:

```bash
brainsurgery flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe_validate.yaml
```

or:

```bash
python3 -m brainsurgery.cli flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe_validate.yaml
```

If the conversion matches, `diff` should report:

```text
Missing on left:
  (none)
Missing on right:
  (none)
Differing:
  (none)
No differences found.
```

That output means the BrainSurgery plan and the reference Python converter
produced equivalent checkpoints.

If you instead see many `model.layers.*.mlp.*` tensors reported as missing on
one side and `model.layers.*.mlp.experts.*` tensors reported as missing on the
other, you are most likely diffing against an older stale output directory from
an earlier partial conversion run.

The reference script can also emit:

- `config.moe_example.json`
  A companion config describing the converted MoE-style checkpoint layout.
- `brainsurgery_conversion.json`
  A small manifest documenting the conversion settings and validation assets.
- `run_olmo_1b_0724_demo.py`
  A one-command runner for the full convert/reference/diff flow.
- `olmo_1b_0724_dense_to_expert_moe_figure.svg`
  A paper-ready vector figure comparing the imperative script, the BrainSurgery
  YAML plan, and the final diff-based validation result.

## Figure Asset

The figure asset is:

```text
flexmore_examples/olmo_1b_0724_dense_to_expert_moe_figure.svg
```

Suggested caption:

```text
Comparison of an imperative dense-to-expert-MoE upcycling script and an equivalent
BrainSurgery workflow. The reference Python implementation mixes tensor-name
mapping, control flow, mutation, and checkpoint writing in handwritten code.
BrainSurgery instead expresses the conversion as a declarative plan consisting of
assertions, tensor copies, initialization, and deletions. The resulting checkpoint
is then validated against the reference implementation using BrainSurgery's built-in
diffing, which reports no missing or differing tensors.
```

## Scope

This example is intentionally a weight-surgery demo. It demonstrates that
BrainSurgery can express and validate the checkpoint rewrite itself.

If you need a runtime-loadable MoE package for a specific framework, you may
still need a framework-specific metadata/config step beyond this demo companion
config.
