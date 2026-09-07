# T1 self-report (condition P)

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution
  succeeded.
- Pitfalls or surprises you hit (one line each):
  - The renumbering collision hazard is only avoided because I built a fresh
    output dict keyed by the new name instead of renaming in place; I also added
    an explicit collision check on the new key.
  - The layer regex has to escape the dots and anchor at `^gpt_neox\.layers\.`,
    otherwise `gpt_neox.final_layer_norm.*` and `embed_out.weight` could be
    caught by a sloppier pattern.
  - Blocks own three non-parameter buffers (`attention.bias`,
    `attention.masked_bias`, `attention.rotary_emb.inv_freq`); matching on
    "the rest of the name" generically rather than enumerating parameter names
    keeps 15 tensors per block instead of 12.
  - `save_file` rejects non-contiguous/shared storage, so I called
    `.contiguous()` on the way out; it was not actually needed here.
  - The output is written only after all checks pass, so a failure leaves no
    partial file.
- Anything in the task text or documentation that was unclear: nothing. The
  explicit old->new index table removed all ambiguity; the `query_key_value`
  interleaved-head layout is described but irrelevant for this task since no
  tensor values are touched.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: a few minutes; one script, one run.
