## Participant self-report

- Final artifact path: `out/T4/plan.yaml`, output written to `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract`/`multiply` require their destination to already exist, so
    building a new "task vector delta" tensor with them isn't possible
    directly; instead the merge was reformulated algebraically as
    `out = base*(1-2*lambda) + lambda*ft1 + lambda*ft2` and built with
    `copy` + `scale_`/`scale` + `add_`, which do allow creating/growing
    scratch tensors.
  - All writes (including scratch tensors) had to stay on the `base::`
    alias, since the output alias is inferred from whichever single alias
    the transforms write to; writing scratch tensors to a separate alias
    would have made the output alias ambiguous.
  - Scratch tensors were named with a leading underscore
    (`base::_tv_acc.<i>.<proj>` etc.) and deleted before the final asserts
    so the output ends up with exactly the original 114 tensor names.
- Anything in the task text or documentation that was unclear: none; the
  README's note on output-alias inference (keep every write on one alias)
  was the key thing to find and follow.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: single pass, a few minutes of
  plan authoring plus one successful run.
