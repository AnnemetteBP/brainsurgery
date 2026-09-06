# Participant self-report: T3 (OLMo-1B-0724-hf), condition B

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `assert count` counts reference matches, not dtypes, so "exactly 112 bfloat16" was expressed as count+dtype on the projection regex plus count+dtype float32 on its negative-lookahead complement.
  - Shard sizes use binary units, so `256MB` in the plan is exactly the required 268,435,456 bytes.
  - Grep tip for the doc pack: the assert operator help is titled `Help for assert.<op>`, not `Help for <op>`.
- Anything in the task text or documentation that was unclear: nothing significant; the README "Output behavior" section answered the sharding questions directly.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
