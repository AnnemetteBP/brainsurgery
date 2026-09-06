# T2 (GPT-2 124M), condition F: participant self-report

- Final artifact path: `out/T2/solution.py` (writes `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `transformers` 5.12.1 has removed `GPT2Model.prune_heads`, so the `prune_heads` route named in `F-allowed.md` for T2 does not exist in this environment.
  - `GPT2Config(n_head=11)` cannot be instantiated (768 is not divisible by 11), and setting `config.pruned_heads` no longer resizes the modules at construction; loading the pruned checkpoint into HF requires replacing each `c_attn`/`c_proj` Conv1D and setting `num_heads=11`, `split_size=704` by hand.
  - `transformers` 5 no longer registers the causal-mask buffer `h.<i>.attn.bias`, so a strict `load_state_dict` reports those twelve keys as unexpected; the checkpoint must still contain them to keep the 160-tensor key set.
- Anything in the task text or documentation that was unclear: nothing; the explicit column/row ranges removed all ambiguity about the head layout.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `safe_open` to read the input and `save_file` to write the output with the original `{"format": "pt"}` metadata. Chosen because a direct tensor-level copy is the only route that guarantees a bit-exact, 160-key output.
  - `torch` 2.14.0: `index_select` to slice the head-bearing axes with an explicit kept-index list (shared between weight and bias so they cannot diverge).
  - `transformers` 5.12.1: used only for post-hoc verification, not for producing the output (see pitfalls for why `prune_heads` was not an option).
  - Not used: `mergekit`, `peft`, `torch-state-bridge`: none of them does intra-tensor head slicing.
- Approximate time spent, if you can tell: about 5 minutes for the solution; roughly the same again for independent verification.

## Verification performed after the run (not part of the attempt count)

- Key set, dtypes and every tensor compared against the input: unchanged tensors are `torch.equal`, and the three pruned tensors per layer equal the input sliced with the exact ranges from TASK.md.
- The output loads into a hand-resized 11-head HF GPT-2 (no missing keys; unexpected keys are exactly the twelve mask buffers) and runs a forward pass with finite logits.
- Its logits agree with the original model with head 5's `c_proj` rows zeroed (mathematically equivalent to removing the head) to a max abs diff of 6.1e-5, with identical argmax.
