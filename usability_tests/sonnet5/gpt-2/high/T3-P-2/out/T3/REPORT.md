## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `h.<i>.attn.c_attn.bias` and the other per-layer bias tensors end with the
    literal substring `attn.bias`, so a naive `endswith("attn.bias")` filter
    (used only while double-checking, not in the final regex) would wrongly
    catch them too; the buffer to drop is the exact name `h.<i>.attn.bias`,
    matched with `^h\.\d+\.attn\.bias$`, not a substring/suffix match.
  - The oversized-tensor rule ("a tensor bigger than the cap goes alone in
    its own shard") falls out for free from a greedy first-fit packer if you
    close the current shard as soon as it exceeds the cap, including a shard
    that started empty — no separate special case needed.
- Anything in the task text or documentation that was unclear: none; the
  tensor list, shapes, and sharding rule were fully specified.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~10 minutes, single script write
  and run, verified bit-exact against the input and re-loaded the sharded
  output to confirm dtypes, dropped buffers, and per-shard byte totals before
  finishing.
