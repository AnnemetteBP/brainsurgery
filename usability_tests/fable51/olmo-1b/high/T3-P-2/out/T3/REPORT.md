# Participant self-report: T3 (OLMo-1B-0724-hf, condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 10 shard files plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The projection matrices were matched with an anchored regex on the exact `model.layers.<i>.(self_attn.{q,k,v,o}_proj|mlp.{gate,up,down}_proj).weight` names rather than `.*weight`, and the resulting set is asserted equal to the expected 112 names so embeddings and `lm_head` cannot be caught.
  - Greedy sharding in sorted name order lands exactly on the 256 MiB budget (14 bfloat16 projection matrices per shard), so the budget check had to be `>` rather than `>=`.
  - `model.embed_tokens.weight` and `lm_head.weight` (412 MB each) exceed the budget and are each stored alone in their own shard.
- Anything in the task text or documentation that was unclear: the task does not say whether the grader checks a specific shard assignment or only the sharding rules; I used greedy packing in sorted key order with HF-style `model-XXXXX-of-XXXXX.safetensors` names.
- Tools used (condition F): not applicable (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: a few minutes.
