# Participant self-report

- **Final artifact path:** `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`)

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  1. `count.of matched zero tensors: base::_d1\..*` — used `assert: { count: { of: ..., is: 0 } }`
     to confirm the scratch tensors were deleted, but `count` treats a zero-match
     reference as an error regardless of the expected `is` value, so it can't be
     used to assert absence.

- **Pitfalls or surprises you hit (one line each):**
  - `count`'s `of` reference errors out on zero matches even when checking `is: 0`;
    switched to `assert: { not: { exists: ... } }` to assert a pattern matches
    nothing, which is the intended idiom.
  - `h.<i>.attn.bias` is a non-trainable causal-mask buffer that lives right next
    to the MLP tensors in the tree dump; the MLP-only regex
    (`h\.\d+\.mlp\.(?:c_fc|c_proj)\.(?:weight|bias)`) had to be scoped tightly
    enough not to accidentally sweep in `attn.bias` or the attention
    `c_attn`/`c_proj` tensors, all of which must stay untouched.
  - With three inputs (`base`, `ft1`, `ft2`), brainsurgery infers the output
    checkpoint from the single alias every transform writes to; doing the merge
    entirely on `base` (using scratch tensor *names* like `base::_d1.<path>`
    rather than a fourth alias) kept the output alias unambiguous, and the
    scratch tensors were deleted before the final tensor-count assert so the
    saved checkpoint has exactly the original 160 names.
  - Confirmed via a throwaway tampered copy of `ft1` (not part of the plan) that
    the shared-tensor `equal` assert does fail loudly on a non-MLP mismatch, and
    independently verified the merged output against a plain NumPy/PyTorch
    recomputation (max relative error `0.0` on the 48 merged tensors, bit-exact
    on the other 112) before finalizing.

- **Anything in the task text or documentation that was unclear:** No. The
  README's note that `equal`'s `right` is "resolved as a rewrite of each `left`
  match, exactly like `to` in copy/move" made it clear the same mechanism could
  be used for both the precondition check and the arithmetic transforms
  (`copy`/`subtract_`/`add_`), which is what let the whole 48-tensor merge be
  expressed as a handful of pattern-based transforms instead of 48 (or 96)
  explicit per-tensor ones.

- **Tools used (condition F):** N/A (condition B).

- **Approximate time spent:** ~20 minutes, most of it reading `docpack/help.txt`
  for exact `add_`/`subtract_`/`scale_`/`assert` semantics and confirming the
  output-alias inference rule before writing the plan.
