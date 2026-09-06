# Participant self-report: T4 (GPT-2 124M, condition P)

- Final artifact path: `out/T4/solution.py` (output checkpoint: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: crash, `FileNotFoundError` on `inputs/base/model.safetensors`; I derived the sandbox root as `Path(__file__).parents[1]`, which is `out/` rather than the sandbox root (the script lives two levels down, in `out/T4/`). Fixed to `parents[2]`.
- Pitfalls or surprises you hit (one line each):
  - Script location `out/<task>/solution.py` is two directories below the sandbox root, so relative-path derivation from `__file__` needs `parents[2]`.
  - `inputs/` is a symlink to an external directory; `Path.resolve()` on the script path is fine, but I intentionally did not resolve the `inputs` symlink itself.
  - Cloned the base tensors before saving so that no tensor in the output dict shares storage with the loaded base dict (safetensors rejects shared storage).
- Anything in the task text or documentation that was unclear:
  - Nothing material. The task is explicit about the 48 MLP names, lambda, dtype and the required checks.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3 only).
- Approximate time spent, if you can tell: about 2 minutes, one failed execution due to the path bug.

Post-write self-check in the script: 160 tensors, 48 merged, 112 non-MLP tensors bit-equal to the base, max relative Frobenius error of merged tensors vs a float64 recomputation 4.0e-08.
