# Participant self-report — T4 (condition P, Pythia-1B)

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard is easy to trip: I kept `base` immutable and computed both task vectors against it inside one expression, rather than accumulating into a running result.
  - `float16` inputs must be lifted to `float32` before the two subtractions/scalings and only cast back at the end, otherwise the rounding eats the small deltas.
  - Bit-exact comparison of the 180 shared tensors needs a NaN-safe test, so I compared flat `uint8` views of the buffers instead of `torch.equal` on floats.
  - Three 2 GB checkpoints do not need to be resident at once: `safe_open` lets each tensor be pulled lazily, so peak memory stays around one checkpoint plus the output dict.
  - The output tensors must be materialised contiguous for `save_file`; results of arithmetic already are, but base tensors are cloned rather than aliased into the writer.
- **Anything in the task text or documentation that was unclear:**
  - The task does not say whether the safetensors `__metadata__` header should be preserved; I copied the base's metadata, since grading only inspects keys/shapes/dtypes/values either choice should be safe.
  - "verify ... every tensor outside the 64 MLP tensors is identical in all three" — I read "identical" as bit-exact, not `allclose`.
- **Tools used (condition F):** n/a — condition P; only `torch` 2.14.0 and `safetensors` 0.5.3.
- **Approximate time spent, if you can tell:** ~5 minutes of authoring; the script itself runs in ~10 s.
