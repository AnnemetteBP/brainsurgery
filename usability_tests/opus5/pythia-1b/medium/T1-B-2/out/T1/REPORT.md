# T1 self-report (Condition B: BrainSurgery plan)

- **Final artifact path:** `out/T1/plan.yaml` -> output `out/T1/model.safetensors`

- **Number of times you executed the script or plan:** 1 execution of the task
  plan (succeeded on the first attempt). Plus 1 execution of a deliberately
  broken copy (`/tmp/neg.yaml`, total-count assert changed to 999, output path
  redirected to `/tmp`) to confirm the required checks really fail loudly.

- **Which executions failed, and why (one line each):**
  - None for the task plan.
  - (Intentional negative test, not a task attempt) `/tmp/neg.yaml`: `failed_assertion` -
    `Error: count failed: model::.* matched 184 tensors, expected 999`; exit code 1 and no
    output file written, which is the behaviour the task requires.

- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collisions: the moves must run in ascending destination order
    (3->2, 4->3, ... 15->11) so each destination is already free; `move` refuses an
    existing destination, so a wrong order would have errored rather than silently
    overwritten, but the delete-first + ascending order avoids the issue entirely.
  - Tensor patterns are full-match regexes, so dots must be escaped
    (`gpt_neox\.layers\.3\.(.*)`) or `layers.1.` would also match `layers.11.`;
    `\d+` in an index position is likewise safe only with escaped dots.
  - The `to` side is a rewrite template with backreferences (`\1`), not a regex,
    so it is written with literal dots.
  - Whole blocks include the three non-parameter buffers (`attention.bias`,
    `attention.masked_bias`, `attention.rotary_emb.inv_freq`); a `.*` tail pattern
    picks them up automatically, giving 15 tensors per block.
  - Output as a single `.safetensors` file needed an explicit filename with the
    suffix; a directory-like path would have produced a sharded directory.

- **Anything in the task text or documentation that was unclear:** Nothing
  blocking. The task text is precise about the old->new index mapping and the
  expected counts. The README's note that a suffix-less output path shards into a
  directory is easy to miss; the QKV interleaved-layout detail is irrelevant for
  this task since blocks move whole and untouched.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent:** ~5 minutes: reading the README and the `move`,
  `delete` and `assert.count` help entries, writing one 40-line plan, running it,
  and verifying the result.
