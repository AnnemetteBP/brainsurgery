# Participant self-report

- Attempts: 2 plan executions; the second succeeded.
- Pitfalls: OmegaConf resolves `${i}` in YAML before BrainSurgery can use it as a structured destination interpolation. The first execution therefore failed during configuration loading; regex capture rewrites (`\1`) fixed the patterned moves.
- Unclear points: The documentation shows structured destination interpolation with `${i}`, but does not call out the OmegaConf escaping needed when that syntax appears in a loaded plan.
- Tools used: BrainSurgery CLI from the provided condition-B environment; `concat` rebuilt the pruned tensors, `delete`/`move` restored the original names, and `assert` enforced the required output invariants.
- Approximate time spent: 5 minutes.
