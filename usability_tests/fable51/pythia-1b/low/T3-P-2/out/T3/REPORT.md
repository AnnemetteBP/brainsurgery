# Participant self-report

- Final artifact path: `out/T3/solution.py` (output: `out/T3/model-0000N-of-00009.safetensors` x9 plus `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Projection matrices are matched with an anchored regex on the four exact module names so `.*weight` cannot overreach onto embeddings, layer norms or the `attention.bias` mask buffer.
  - The two embeddings are 412 MB in float32 (the task text quotes their float16 size), so each is placed alone in its own shard as required.
- Anything in the task text or documentation that was unclear: the "206 MB each" figure refers to the float16 input size, not the float32 output size; harmless since both exceed the shard budget.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
