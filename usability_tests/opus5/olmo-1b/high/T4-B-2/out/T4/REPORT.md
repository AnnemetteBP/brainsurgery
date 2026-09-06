# T4 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T4/model.safetensors` (114 tensors, float32),
  produced by `out/T4/plan.yaml` via `brainsurgery out/T4/plan.yaml`.
  Run log: `out/T4/run1.log`.

- **Number of times you executed the script or plan:** 1 execution of
  `out/T4/plan.yaml` (succeeded first time, 33 s wall clock).
  Before that I ran 3 executions of throwaway probe plans on the 8 MB
  `inputs/lora/adapter_model.safetensors` to check syntax and semantics
  against a cheap input instead of a 15 GB load
  (`out/_probe/probe.yaml`, `probe2.yaml`, `probe3.yaml`).

- **Which executions failed, and why (one line each):**
  - `out/T4/plan.yaml` execution 1: did not fail.
  - Probe 3 (`out/_probe/probe3.yaml`) failed on purpose: `count` asserted 63
    where 64 tensors matched, to confirm a failing assert aborts the run with
    exit code 1 (`TransformError: count failed: ... matched 64 tensors, expected 63`).

- **Pitfalls or surprises you hit (one line each):**
  - Alias inference for the output: any write (including in-place `scale_` /
    `add_` / `subtract_` and `copy` destinations) has to land on a single alias,
    so both task vectors had to be built as temporaries *inside* the `base`
    alias (`base::tv1.<name>`) rather than by scaling `ft1::` / `ft2::` in place.
  - The ordering hazard is real and easy to get wrong: `tv1` and `tv2` are both
    materialised and scaled *before* either `add_` touches `base`, so each task
    vector is taken against the unmodified base.
  - There is no name-only comparison operator, so "the three checkpoints have
    the same tensor names" is expressed as `equal` with `eps: 1.0e+30`
    (which still enforces the name mapping plus shape and dtype, but accepts
    any values) combined with `count: 114` on each alias; exact `equal` with no
    eps is then used for the 66 non-MLP tensors.
  - YAML gotcha: `eps: 1e30` parses as a *string* in PyYAML; it must be written
    `1.0e+30`.
  - References are full-match regexes, which is what makes the temporaries safe:
    `tv1.model.layers.0.mlp.gate_proj.weight` does not match
    `model\.layers\.\d+\.mlp\..*`, so the in-place steps never hit them.
  - Capture-group numbering: the MLP pattern needs a *non-capturing*
    `(?:gate_proj|up_proj|down_proj)` inside the outer group so that `\1` in the
    `to` template is the whole tensor name.
  - Peak memory is ~21 GB (three 5 GB checkpoints plus two 3.2 GB task-vector
    sets); the default `inmemory` provider handled it, no `--provider arena`
    needed.

- **Anything in the task text or documentation that was unclear:**
  - "verify that the three checkpoints have the same tensor names" has no direct
    operator in the tool; the eps trick above is a workaround, and the README
    does not say whether `equal` is meant to be used that way.
  - The task says the merge must be "computed in float32" but does not say
    whether the closed form `0.2*base + 0.4*ft1 + 0.4*ft2` is acceptable; I used
    the literal task-vector form instead, which also makes the ordering explicit.
  - The doc pack README contains absolute paths from the author's machine
    (`/Users/petersk/...`) in the documentation links, which are dead here.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `help.txt` and running the cheap probe plans.

## Independent check of the result (not part of the plan)

Read back with `safetensors` + `numpy`: key set identical to the base index
(114), all shapes/dtypes preserved as float32, the 66 non-MLP tensors bit-exact
against base *and* ft1 *and* ft2, and for the 48 MLP tensors the worst relative
Frobenius error against `base + 0.4*(ft1-base) + 0.4*(ft2-base)` computed in
float64 is 3.5e-08 (limit 1e-5).
