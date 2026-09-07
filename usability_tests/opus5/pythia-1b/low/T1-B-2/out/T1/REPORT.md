# T1 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T1/plan.yaml` -> `out/T1/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: `move` refuses existing destinations, so the ten renames are ordered ascending by new index (3->2, 4->3, ... 15->11) after the deletions, which guarantees each destination slot is already free.
  - Regex references are full-match, so dots must be escaped (`gpt_neox\.layers\.3\.(.*)`) to avoid matching e.g. layer 13 or unrelated names; the `to` side uses the literal name plus `\1`.
- Anything in the task text or documentation that was unclear:
  - The README documents `assert: count` but not whether `of` counts across the whole state dict; I relied on `count: { of: '.*', is: 184 }` for the total-tensor check and it behaved as expected.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes.

## Plan structure

1. `assert count` 16 blocks in the input.
2. `delete` every tensor of blocks 2, 6, 10, 14 (`gpt_neox\.layers\.(2|6|10|14)\..*`).
3. `assert count` 12 blocks remain.
4. Ten `move` transforms renumbering survivors in ascending target order.
5. Required checks: `assert not exists` for blocks 12-15, `assert count` 12 `query_key_value.weight`, `assert count` 180 block tensors, `assert count` 184 tensors total.
6. `output` to `out/T1/model.safetensors`, format safetensors.
