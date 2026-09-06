# T1 (Pythia-1B, condition B) — participant self-report

- **Final artifact path:** `out/T1/plan.yaml` → `out/T1/model.safetensors`
  (184 tensors, float16 + uint8 buffers).

- **Number of times you executed the script or plan:** 1 execution of
  `out/T1/plan.yaml` (passed on the first attempt). Two additional executions
  were negative controls on *copies* of the plan, written to
  `out/_negctl/` (since deleted) and never to `out/T1`:
  one with the required 184-tensor check flipped to 183, one with the
  `layers.15 -> layers.11` move removed.

- **Which executions failed, and why (one line each):**
  - The real plan did not fail. `out/T1/plan.yaml` run #1: exit 0, output written.
  - Negative control 1 (`failed_assertion`, intentional): `count failed: model::.* matched 184 tensors, expected 183` — exit 1, no output file written.
  - Negative control 2 (`failed_assertion`, intentional): dropping the last renumbering move left block 15 in place and tripped `not: { exists: 'gpt_neox\.layers\.1[2-5]\..*' }` — exit 1, no output file written.

- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collisions were the real hazard: every destination index is lower than its source, so processing sources in **ascending** order guarantees each destination slot is already free; `move` refusing an existing destination is the safety net, so a wrong order aborts instead of silently clobbering a surviving block.
  - Deleting the dropped blocks *before* renumbering is what frees slots 2, 6 and 10; doing the moves first would collide on old index 2.
  - Regex is full-match, so `gpt_neox\.layers\.(?:2|6|10|14)\..*` does not leak onto `layers.12`/`layers.14x`; the trailing escaped dot is what makes `2` safe against `12`. Unescaped dots would have been a silent overmatch.
  - The 15 tensors per block include three non-parameter buffers (`attention.bias` uint8, `attention.masked_bias`, `attention.rotary_emb.inv_freq`); a pattern targeting only `weight`/`bias` parameters would have left them behind. Targeting the whole `layers.<i>.` prefix handles them for free.
  - Structural counts (244 in, 184 out, 15 tensors in each of blocks 0..11, 180 block tensors inside indices 0..11) pin down the result far better than the three required checks alone: the required checks would still pass if a block were duplicated at the wrong index, the per-index counts would not.
  - The `-> [out, in]` layout, the fused interleaved `query_key_value` layout, and the float16 storage never mattered here — this task is a pure rename, so no tensor is ever read or rewritten, only rebound.

- **Anything in the task text or documentation that was unclear:** Nothing
  blocking. Two small things: the README does not spell out that `to` in
  `move` is a regex *replacement template* (so `\1` works and `.` is literal
  there) — I inferred it from the `assert: equal` docs, which do say it; and
  it is not stated whether `assert: count` accepts `is: 0`, so I used
  `not: { exists: ... }` for the "nothing remains" checks instead.

- **Tools used (condition F):** n/a — condition B, plan only.

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `docpack/help.txt` and verifying the result; the plan run itself takes ~7 s.
