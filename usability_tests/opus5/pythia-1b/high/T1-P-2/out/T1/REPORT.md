# T1 — Participant self-report (Condition P)

- **Final artifact path:** `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The renumbering-collision hazard is avoided by construction: I built a fresh
    output dict keyed by new names instead of renaming in place, and added an
    explicit collision check on insert, so an in-place shift could never clobber
    a survivor.
  - The layer regex has to anchor and escape the dots (`^gpt_neox\.layers\.(\d+)\.(.+)$`);
    an unanchored/unescaped pattern would be an overmatch risk, and `\d+` (not `\d`)
    matters for two-digit indices 10..15.
  - I compared tensors through raw bytes rather than `torch.equal`, so NaN entries
    (possible in fp16 weights or the `masked_bias` buffer) still compare equal
    bit-for-bit; a naive `==` check would have spuriously "failed loudly".
  - `attention.bias` (uint8 causal mask), `attention.masked_bias` (0-dim scalar) and
    `rotary_emb.inv_freq` are non-parameter buffers but still per-block tensors, so
    they must be dropped/renumbered with the rest — I asserted 15 tensors per block
    to catch missing any of them.
  - All checks run on the in-memory dict *before* `save_file`, so a failure leaves
    no partial output on disk.
- **Anything in the task text or documentation that was unclear:**
  - Nothing blocking. The task describes the fused `query_key_value` head-interleaved
    layout in detail, which is irrelevant for a pure block drop/rename — I did not
    need to touch tensor contents at all. Also unstated: whether the HF `config.json`
    should be rewritten to `num_hidden_layers: 12`; the "Required result" asks only
    for the single `.safetensors` file, so I wrote only that.
- **Tools used (condition F):** n/a (condition P: Python + torch 2.14.0 + safetensors 0.5.3)
- **Approximate time spent, if you can tell:** ~5 minutes; the script itself runs in ~7 s.
