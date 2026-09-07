# T1 run record — Condition P (Python / PyTorch)

## Participant self-report

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the first
  execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collision hazard: shifting blocks in place would let old 3 land
    on surviving old 2's slot; I built a fresh output dict keyed by the new
    names instead, and still assert that no new key is written twice.
  - The block regex has to be anchored (`^h\.(\d+)\.`) with escaped dots, or a
    loose pattern would also touch `mlp.c_proj` / `attn.c_proj` name fragments
    and the 4 non-block tensors.
  - Each block owns 13 tensors, not 12: the causal-mask buffer `attn.bias` is
    stored in the checkpoint and has to move with its block (12*13+4 = 160 in,
    9*13+4 = 121 out).
  - `attn.bias` vs `attn.c_attn.bias` are easy to conflate; matching on the
    full suffix rather than a substring avoids it.
  - No transposes or dtype casts were needed here, so GPT-2's Conv1D `[in, out]`
    layout never came into play — the task is a pure rename/delete.
- **Anything in the task text or documentation that was unclear:** nothing
  blocking. Two small ambiguities I resolved by reading "Required result" 4
  literally: the output directory holds only `model.safetensors` (no config or
  tokenizer copies), and the "no tensor of blocks 9, 10, 11 remains" check is
  about the *output* index space (only 0..8 may appear), not about the source
  blocks that happen to be numbered 9-11 and do survive under new names 6-8.
- **Tools used (condition F):** n/a — condition P.
- **Approximate time spent, if you can tell:** ~5 minutes.

## Notes on the implementation

The script fails loudly (`SystemExit(1)`, nothing written) on any of:

- input not shaped as expected (not 160 tensors, blocks not 0..11, a block
  without exactly 13 tensors, non-block set != the 4 expected names);
- a renumbering collision;
- any output block index >= 9;
- a count of `h.<i>.attn.c_attn.weight` matches != 9;
- output block indices not contiguous 0..8;
- output tensor count != 121;
- a shape/dtype change, or a tensor that is not the original object.

All checks run before writing; the file is written to a `.tmp` path and
`os.replace`d, so a failure can never leave a partial output. After writing,
the file is reloaded and every tensor is compared bit-exactly (`torch.equal`)
plus shape/dtype against the source.
