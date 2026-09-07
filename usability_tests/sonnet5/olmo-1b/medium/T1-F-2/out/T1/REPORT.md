# T1 Participant self-report

- **Final artifact path:** `out/T1/solution.py` (invoked via `out/T1/run.sh`), producing `out/T1/model.safetensors`.
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - None of real substance. Building the output as a brand-new dict keyed by target layer index (rather than renaming in place) sidesteps the renumbering-collision hazard by construction, since old and new key spaces are never mixed.
  - safetensors requires contiguous, non-shared tensor storage, so tensors are `.contiguous().clone()`d before saving even though nothing here shares memory.
- **Anything in the task text or documentation that was unclear:** No.
- **Tools used (condition F): name, version, and why:** `safetensors` 0.5.3 (`safetensors.torch.load_file`/`save_file`) to read the sharded checkpoint and write the single-file output, and `torch` 2.14.0 for tensor operations. Chose a plain script over `mergekit`/`torch-state-bridge` because the required transform (drop 4 whole blocks, remap the rest into a fresh contiguous namespace) is a direct dict rebuild keyed off `model.safetensors.index.json`, with no need for a merge config or regex-capture rewrite engine; a script also makes it straightforward to embed the required checks so the run fails loudly and writes nothing on failure.
- **Approximate time spent, if you can tell:** A few minutes.
