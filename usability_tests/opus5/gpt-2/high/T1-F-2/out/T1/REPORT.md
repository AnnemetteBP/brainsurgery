# T1 — Depth pruning with layer renumbering (GPT-2 124M) — Condition F

- **Final artifact path:** `out/T1/solution.py` (runner: `out/T1/run.sh`, output: `out/T1/model.safetensors`)

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  - None. Execution 1 produced the correct 121-tensor output; execution 2 was a
    re-run after I *strengthened a check* (see below), and also succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - `torch_state_bridge.RuleEngine` applies its rules **sequentially** to every key
    (`for r in self.rules: key = r.regex.sub(...)`), so a naive one-phase
    `h.<old>. -> h.<new>.` rule set can cascade a rename into a later rule and collide;
    I renamed through a marker namespace (`h.<old>. -> h.#<new>#. -> h.<new>.`) so
    rule order cannot matter.
  - The library's `{n}` captures are digits-only and its literal parts are
    `re.escape`d, so dots are not a wildcard hazard here — but there is no anchoring,
    so a literal rule `h.1.` is only safe because no key contains `h.1.` as an
    inner substring; the marker phase removes that dependence too.
  - The causal-mask buffer `h.<i>.attn.bias` is a block tensor and must be renumbered
    like the rest; it is easy to mistake for a bias of `attn.c_proj`/`c_attn`.
  - **A deliberate negative test showed the three "Required checks" are purely
    structural**: dropping blocks 0,1,2 instead of 2,5,8 still yields 9 contiguous
    blocks and 121 tensors and passes all three. I added a check pinning the
    kept-block mapping to the one the task specifies, then re-ran (execution 2).
  - No shared-storage/tied-tensor problem in this checkpoint (`lm_head` is absent),
    so `save_file` needed no de-duplication; I still call `.contiguous()`.

- **Anything in the task text or documentation that was unclear:**
  - "Output: a single file `out/T1/model.safetensors`" vs. grading that "compares
    `out/T1` with a hidden reference" — I read this as comparing the tensors only, so
    I wrote no `config.json`/tokenizer copies, even though a 9-layer checkpoint would
    need `n_layer: 9` to actually load into `GPT2Model`.
  - The three required checks do not pin *which* blocks were removed (see above);
    I treated them as a minimum, not as sufficient.

- **Tools used (condition F): name, version, and why:**
  - `torch-state-bridge` 0.1.0 — the rule-based key rewriting itself. It is the
    condition-F package aimed at exactly this (regex/capture key renaming), and I
    used its `parse_rules`/`RuleEngine` for both rename phases. Its value here was
    modest: the rules are a 2-line generator either way, and its sequential-apply
    semantics were themselves a collision hazard I had to design around; it has a
    `state_bridge_preview`/`print_diff` surface but no collision detection on the
    path I used, so the collision guard is mine.
  - `safetensors` 0.5.3 — `load_file` / `save_file`; the input and required output
    are both single safetensors files, so this is the direct I/O route.
  - `torch` 2.14.0 — tensor dtype/shape/value identity checks and `torch.equal`
    in the post-write read-back verification.
  - **Considered and rejected:** `mergekit` 0.1.4 layer slicing (`passthrough`) is the
    advertised route for depth pruning, but it operates on HuggingFace *model
    directories* through `transformers`, which would (a) require an edited 9-layer
    `config.json`, (b) round-trip keys through `GPT2LMHeadModel` and re-emit them with
    a `transformer.` prefix and a `lm_head.weight`, and (c) drop or regenerate the
    `attn.bias` mask buffers — i.e. it cannot produce this exact 121-key set bit-exactly
    without post-processing that is larger than the task itself. `transformers`
    `save_pretrained` has the same key-set problem. `peft` is unrelated here.

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `torch_state_bridge`'s source to learn that rules chain sequentially.
