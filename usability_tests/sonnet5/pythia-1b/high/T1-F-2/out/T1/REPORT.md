# Participant self-report — T1 (Pythia-1B, condition F)

- **Final artifact path:** `out/T1/solution.py` (invoked via `out/T1/run.sh`),
  output at `out/T1/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The renumbering has a collision hazard if blocks are shifted in the wrong
    order (e.g. processing indices out of order, or writing new keys before
    all old keys are read); I avoided it by computing the full
    old-index -> new-index mapping up front from the sorted list of
    surviving indices, and by asserting no output key is produced twice.
  - The GPT-NeoX interleaved QKV layout (768-row per-head blocks, further
    split into contiguous 256-row q/k/v segments) is a red herring for this
    task: T1 only removes and renumbers whole blocks, no row-level surgery is
    needed, so it didn't need to be touched — but it was worth checking that
    no step here changes tensor contents, only key names.
- **Anything in the task text or documentation that was unclear:** no,
  the exact old->new index mapping was given explicitly in the task, which
  made verification straightforward.
- **Tools used (condition F):** `safetensors` 0.5.3 only (`safe_open`,
  `save_file`), plus Python's standard `re`. I did not use mergekit or
  torch-state-bridge: this is a one-shot rule-based rename/drop over a single
  safetensors file with no arithmetic or merging involved, so a ~90-line
  plain script expressing the exact rule (drop 4 named blocks, remap the rest
  via an explicit old->new dict, copy 4 non-block tensors unchanged) is more
  auditable and has fewer moving parts than routing it through a general
  merge-config YAML or a separate regex-rewrite library, and it makes the
  required checks (no blocks 12-15, exactly 12 blocks, exactly 184 tensors)
  trivial to embed directly as assertions that abort before any write.
- **Approximate time spent:** ~10 minutes (single script, single run, no
  retries needed).
