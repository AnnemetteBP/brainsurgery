# T4 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T4/plan.yaml` (output checkpoint at
  `out/T4/model.safetensors`, 244 tensors, 228 float16 + 16 uint8).

- **Number of times you executed the script or plan:** 2 executions of
  `out/T4/plan.yaml` (1 failed, then 1 success). Separately I ran three
  read-only scratch plans that are not the solution and write no output, kept
  under `out/T4/scratch/`: `inspect.yaml` (validate the assertion forms before
  spending an attempt), `verify.yaml` and `measure.yaml` (cross-check the
  written checkpoint against a recomputation using the literal formula).

- **Which executions failed, and why (one line each):**
  1. `failed_assertion` — `equal failed: base::gpt_neox.layers.0.attention.masked_bias != ft1::...`:
     I used `equal` with a huge `eps` as a name/shape/dtype parity check over all
     244 tensors, but `masked_bias` is `-inf` in float16, so `|(-inf) - (-inf)|`
     is `NaN` and `NaN <= eps` is false. Fixed by restricting the eps-based
     parity check to the 64 (finite) MLP tensors and comparing the other 180
     exactly, where `equal` uses `torch.equal` and handles infinities.
  2. — (second execution passed).

- **Pitfalls or surprises you hit (one line each):**
  - `equal` with `eps` is not a superset of `equal` without `eps`: the eps path
    goes through `abs(a - b) <= eps`, which is false for `inf` vs `inf`, while
    the exact path uses `torch.equal`, which is true. GPT-NeoX checkpoints carry
    `attention.masked_bias = -inf` in float16, so this bites immediately.
  - There is no "assert these two aliases have the same tensor names" operator.
    I built it out of `count` (244 and 64 per alias) plus `equal` with a rewrite
    right-hand side, which requires every `left` match to resolve to an existing
    `right` tensor of the same shape and dtype; the two `equal` checks together
    cover all 244 names, and the counts close the argument that the name sets
    are identical rather than merely overlapping.
  - `cast_` takes the target dtype under the key `to`, not `dtype` like `cast`.
  - Regex destinations are `re.sub`, not an anchored `expand`, so a `from`
    pattern that only matches a substring would silently rewrite part of a name.
    Anchoring on the `gpt_neox.` prefix and using `\1` around the whole match
    avoids it.
  - The arithmetic hazard in the task (each task vector must be taken against
    the *unmodified* base) disappears if you expand the formula:
    `base + L*(ft1-base) + L*(ft2-base) = (1-2L)*base + L*ft1 + L*ft2`, i.e.
    `0.2*base + 0.4*ft1 + 0.4*ft2`. `ft1` and `ft2` are then read-only and the
    accumulation happens in a float32 copy of the base MLP tensors, so no
    intermediate can contaminate a later term. I confirmed the rearrangement
    against the literal three-term formula (see below).
  - Only one alias may be written to if `output` is set, so every destination
    (including the float32 scratch tensors from `ft1`/`ft2`) has to be an
    explicit `base::` reference.
  - Output path with a `.safetensors` suffix writes a single file; the default
    `--shard-size` only applies to directory-style output paths.
  - The MLP scratch tensors must be `delete`d before the output is written, or
    the checkpoint would have 372 tensors instead of 244.

- **Anything in the task text or documentation that was unclear:**
  - "Verify that the three checkpoints have the same tensor names" has no direct
    operator; it is only reachable indirectly, and how to build it (counts plus
    a rewrite-based `equal`) took the most thought of anything in the task.
  - The README documents `equal`'s `eps` as "absolute tolerance" but not that it
    changes the comparison path and therefore the handling of non-finite values.
  - "computed in float32, then cast back to float16" does not say whether an
    algebraically equivalent regrouping is acceptable; the 1e-3 relative
    Frobenius tolerance implies it is, and I verified it empirically.

- **Tools used (condition F):** n/a.

- **Approximate time spent, if you can tell:** roughly 25 minutes, most of it
  reading `help.txt` and the installed `brainsurgery` sources (`add.py`,
  `cast.py`, `name_mapping.py`, `output_paths.py`, `equal.py`) to pin down
  regex-rewrite and output-alias semantics before the first execution.

## Checks in the plan

Implemented as `assert` transforms, in order:

1. Before anything else: `count` 244 and `count` 64 (MLP pattern) on each of
   `base`, `ft1`, `ft2`; `equal` with `eps: 1.0e+30` mapping the 64 base MLP
   names onto `ft1`/`ft2` (name, shape and dtype parity); `equal` without `eps`
   on the 180 non-MLP tensors against both fine-tunes (bit-identical backbone).
2. `count: { of: 'base::tv1\..+', is: 64 }` and the same for `tv2` — exactly 64
   task-vector tensors per fine-tune were built and added.
3. After the merge: `count` 244 on the output alias, `count` 64 on the MLP
   pattern, `not exists` for leftover scratch tensors, `dtype: float16` on the
   merged tensors, and a repeat of the non-MLP exact `equal` against both
   fine-tunes, which proves the other 180 tensors still hold the base values.

## Independent verification (scratch, outside the solution plan)

`scratch/verify.yaml` recomputes the merge with the task's literal formula and a
different arithmetic order — `d1 = 0.4*(ft1-base)`, `d2 = 0.4*(ft2-base)`, both
against the untouched base, then `base + d1 + d2` — and compares it to the
written checkpoint. The 180 non-MLP tensors and all 32 MLP bias tensors are
bit-identical; the MLP weights differ on a few elements by at most 3.8e-6
absolute (one float16 ULP at those magnitudes, weights ranging over ~0.13-0.93),
with a mean error of ~2e-12. `scratch/measure.yaml` asserts this bound
(`iszero` with `eps: 0.0005`) and dumps the statistics. That is far inside the
1e-3 relative Frobenius tolerance.
