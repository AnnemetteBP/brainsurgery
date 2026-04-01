# FlexMore Examples

This directory now contains paper-ready OLMo-1B checkpoint surgery examples for
both dense-to-expert-MoE upcycling and PHLoRA-based FlexMoRE-style expert
compression.

The dense-to-expert-MoE workflow has three complementary pieces:

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

The PHLoRA companion example is:

4. `olmo_1b_0724_hf_moe_to_flexmore_phlora.yaml`
   BrainSurgery conversion plan that starts from the 2-expert MoE checkpoint
   produced by the previous example and rewrites expert 1 into PHLoRA delta
   factors relative to expert 0 by:
   - copying each expert-1 MLP matrix into an explicit temporary delta tensor
   - subtracting the matching expert-0 matrix to form `expert_1 - expert_0`
   - factorizing those deltas with `phlora`
   - deleting the original dense expert-1 matrices to leave a dense-anchor +
     low-rank-delta layout

   This is the checkpoint-surgery core of a heterogeneous FlexMoRE-style model:
   expert 0 stays dense, while expert 1 is represented as a low-rank adaptor.
   A framework-specific runtime or metadata layer is still responsible for
   interpreting the generated `phlora_a/phlora_b` tensors during execution.
   Reference + validation companions:
   - `olmo_1b_0724_hf_moe_to_flexmore_phlora_reference.py`
   - `olmo_1b_0724_hf_moe_to_flexmore_phlora_validate.yaml`

5. `olmo_1b_0724_hf_moe_to_low_rank_expert.yaml`
   More literal companion to `low_rank_expert.py`. This plan keeps the output
   as a dense 2-expert MoE checkpoint, but rewrites expert 1 in place as a
   rank-limited approximation around expert 0 by:
   - subtracting expert 0 from expert 1
   - applying `phlora_` to the delta in place
   - adding expert 0 back

   This is the closest declarative BrainSurgery analogue to the imperative
   "dense expert plus low-rank correction" flow in the Python script.
   Reference + validation companions:
   - `olmo_1b_0724_hf_moe_to_low_rank_expert_reference.py`
   - `olmo_1b_0724_hf_moe_to_low_rank_expert_validate.yaml`

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

After that MoE upcycling step, you can also derive the complementary
FlexMoRE-style PHLoRA checkpoint:

```bash
brainsurgery flexmore_examples/olmo_1b_0724_hf_moe_to_flexmore_phlora.yaml
```

This produces a checkpoint where expert 0 remains dense and expert 1 is stored
as PHLoRA delta factors relative to expert 0. It is a useful paper/demo example
for showing that BrainSurgery can express not only expert creation, but also
heterogeneous expert compression.

Build the matching reference output:

```bash
python3 flexmore_examples/olmo_1b_0724_hf_moe_to_flexmore_phlora_reference.py \
  --model models/olmo_1b_0724_hf_dense_moe_demo \
  --target models/olmo_1b_0724_hf_flexmore_phlora_r64_reference \
  --copy-metadata \
  --write-example-config
```

Diff the YAML and reference outputs:

```bash
brainsurgery flexmore_examples/olmo_1b_0724_hf_moe_to_flexmore_phlora_validate.yaml
```

If you want the output to remain a dense MoE checkpoint while still compressing
expert 1 through a rank-limited delta approximation, run:

```bash
brainsurgery flexmore_examples/olmo_1b_0724_hf_moe_to_low_rank_expert.yaml
```

Build the matching reference output:

```bash
python3 flexmore_examples/olmo_1b_0724_hf_moe_to_low_rank_expert_reference.py \
  --model models/olmo_1b_0724_hf_dense_moe_demo \
  --target models/olmo_1b_0724_hf_low_rank_expert_r64_reference \
  --copy-metadata \
  --write-example-config
```

Diff the YAML and reference outputs:

```bash
brainsurgery flexmore_examples/olmo_1b_0724_hf_moe_to_low_rank_expert_validate.yaml
```

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
- `olmo_1b_0724_flexmore_validation_figure.svg`
  A paper-ready vector figure tying the MoE upcycling workflow to the two
  FlexMoRE companion conversions and a SYNAPSE-level architectural bridge.

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

Additional figure asset:

```text
flexmore_examples/olmo_1b_0724_flexmore_validation_figure.svg
```

Suggested caption:

```text
BrainSurgery provides a cohesive workflow from checkpoint upcycling to validated
FlexMoRE-style compression. First, two dense OLMo checkpoints are rewritten into
a validated 2-expert MoE checkpoint. From that shared intermediate, BrainSurgery
then expresses two complementary compression strategies: a factorized PHLoRA
layout that stores expert deltas explicitly, and an in-place low-rank rewrite
that preserves a dense expert tensor interface. In both cases, declarative YAML
plans are paired with independent Python references and diff-based validation.
This checkpoint-surgery workflow also provides a natural bridge to SYNAPSE, where
the resulting tensor layouts can be connected to explicit executable model
structure.
```

## Scope

This example is intentionally a weight-surgery demo. It demonstrates that
BrainSurgery can express and validate the checkpoint rewrite itself.

If you need a runtime-loadable MoE package for a specific framework, you may
still need a framework-specific metadata/config step beyond this demo companion
config.
