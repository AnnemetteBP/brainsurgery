# T4 (condition F) — participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  - Execution 1: `crash` — `RuntimeError: self.dim() cannot be 0 to view Half as Byte`; my
    bit-exact byte comparison re-viewed each tensor as `uint8`, which is illegal on the
    0-dim scalar tensor present among the 180 non-MLP tensors. Fixed by `reshape(-1)`
    before the view. No output had been written at that point.
- **Pitfalls or surprises you hit (one line each):**
  - The checkpoints are not uniformly float16: some non-MLP tensors are `U8`/bool
    (`gpt_neox.layers.<i>.attention.bias` causal masks) and at least one is 0-dim, so the
    verification had to be dtype- and rank-agnostic rather than assume `float16`.
  - The ordering hazard is real but easy to defuse: I read `base` straight from the file
    for every tensor and never write back into it, so both task vectors are necessarily
    taken against the pristine base.
  - Bit-exactness for the 180 untouched tensors is only guaranteed if they are *copied*,
    not recomputed; a "merge everything, the deltas are zero" route would round-trip them
    through float32 and, for the U8 mask buffers, through a dtype cast as well.
  - fp32 accumulate then cast back to fp16 gives a worst-case relative Frobenius error of
    2.4e-04 over the 64 merged tensors, comfortably inside the 1e-3 tolerance.
- **Anything in the task text or documentation that was unclear:**
  - The task says the base is "244 tensors, float16", but the file also contains `U8`
    tensors; that is only a problem if a solution keys its logic on the stated dtype.
  - "Abort with an error" does not say whether the check must be on the *bytes* or on
    numeric equality; I chose byte-exact, which is the stricter reading and also matches
    the grader's bit-exact requirement for the unchanged tensors.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — streaming per-tensor read via `safe_open` (keeps peak memory at
    a couple of tensors instead of three full 2 GB checkpoints) and `save_file` for a
    single unsharded `model.safetensors`.
  - `torch` 2.14.0+cu130 — float32 arithmetic, dtype casts, and the byte-level comparison.
  - Why not `mergekit` 0.1.4 (the suggested `task_arithmetic` route): its merge method
    applies to *every* tensor, so the 180 untouched tensors would be recomputed rather
    than copied, putting the grader's bit-exact requirement at the mercy of mergekit's
    dtype and out-dtype handling; it writes a sharded HF model directory with an index
    rather than the single `out/T4/model.safetensors` the task demands; and it has no way
    to express the required checks (shared-tensor verification, exactly 64 merged, exactly
    244 out), so a wrapper script enforcing them would have been needed regardless. A
    ~150-line script does the whole job with the checks inline.
  - Why not `transformers`/`peft`: nothing here needs a model graph or an adapter; loading
    `GPTNeoXForCausalLM` three times would only risk tied-weight and dtype rewrites on
    tensors that must survive bit-exact.
- **Approximate time spent, if you can tell:** ~10 minutes, of which ~20 s is the merge run
  itself.

## What the script enforces (run fails loudly if any does not hold)

1. All three checkpoints expose the same 244 tensor names, and the base has exactly 244.
2. The 64 expected MLP names all exist; the complement is exactly 180 tensors.
3. Every one of those 180 tensors is **byte-identical** in base, ft1 and ft2 (the
   frozen-backbone precondition), reported with the offending names if not.
4. Per merged tensor, shape and dtype agree across the three checkpoints.
5. Exactly 64 tensors were merged; the output dict holds exactly 244 tensors.
6. After writing, the file is re-opened and checked: 244 tensors, key set equal to the
   base's, and every tensor byte-identical to what was intended.

All checks raise `CheckFailed` and exit non-zero with a `CHECK FAILED:` message.
