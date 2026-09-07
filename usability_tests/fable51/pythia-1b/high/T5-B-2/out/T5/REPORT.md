# Participant self-report: T5 (Pythia-1B, condition B)

- Final artifact path: `out/T5/plan.yaml` (executed summary in `out/T5/summary.yaml`; output shards and `model.safetensors.index.json` in `out/T5/`)
- Number of times you executed the script or plan: 1 execution of `out/T5/plan.yaml` (succeeded first time). Before that, 3 executions of a throwaway probe plan (`out/scratch/probe.yaml`, deleted afterwards) that loaded only the 8 MB adapter, had no `output`, and only tested the `matmul` name-mapping syntax.
- Which executions failed, and why (one line each):
  - probe run 1: `checkpoint path does not exist` because I ran it from `out/scratch/`; input paths resolve relative to the current working directory, not the plan file.
  - probe run 2: `matmul source_b missing` because I wrote `from_b` as an escaped regex (`\.`); `from_b` is rewritten from `from_a` captures and then looked up as a literal name, like `to`.
  - probe run 3: passed. The real plan then ran once and passed.
- Pitfalls or surprises you hit (one line each):
  - `from_b` in `matmul` (and `from` in `add_`) is a rewrite of the `from_a`/`to` match with `\1`, not an independent regex; the help text does not say this explicitly (only `assert equal` documents it).
  - With two inputs the output alias is inferred from where transforms write, so the intermediate `B @ A` product had to be created on the `base` alias and then deleted there; creating it on `lora` would have made the output alias ambiguous.
  - `add_` of a float32 delta into a float16 target: rather than rely on implicit promotion I cast the 16 targets to float32, added, then cast back to float16, which also matches the "computed in float32" requirement.
  - Shard sizes are binary units, so `512MB` in the plan equals the required 536,870,912 bytes.
- Anything in the task text or documentation that was unclear:
  - TASK.md says the two 206 MB embedding tensors are "larger than" the 512 MiB budget and stored alone; they are not, and the tool packs them greedily with other tensors (shard 1 holds 24 tensors). I followed the tool's greedy rule, which is what the grader's stated sharding rules describe.
  - The README `add` example (`from_a: '.*.weight', from_b: '.*.delta', to: '.*.weight'`) suggests independent pattern matching for binary ops, which is not how it resolves; a sentence in the `matmul`/`add`/`add_` help about capture rewriting would have saved the probe run.
- Tools used (condition F): n/a (condition B). A read-only Python snippet was used after the run to verify the output against a direct computation; it made no edits.
- Approximate time spent, if you can tell: about 5 minutes.
