# brainsurgery
Swiss army knife for scripted tensor surgery on model checkpoints.

## Installation
```bash
pip install brainsurgery
```

## Quick start
Run a YAML plan:

```bash
brainsurgery examples/gpt2.yaml
```

Run multiple config inputs (deep-merged in order), then CLI overrides:

```bash
brainsurgery base.yaml override.yaml transforms[0].dump.format=compact
```

Enable interactive mode after configured transforms:

```bash
brainsurgery -i examples/gpt2.yaml
```

## CLI
`brainsurgery [OPTIONS] [CONFIG_ITEMS]...`

`CONFIG_ITEMS` can be:
- YAML files (`.yaml`, `.yml`) loaded and deep-merged in order.
- `key=value` overrides applied last (supports dotted keys and list indices like `transforms[0].dump.format=tree`).

Options:
- `--shard-size TEXT` (default: `5GB`) default shard size for directory outputs.
- `--num-workers INTEGER` (default: `8`) max parallel I/O workers.
- `--provider TEXT` (default: `inmemory`) one of `inmemory`, `arena`.
- `--arena-root PATH` (default: `.brainsurgery`) arena storage root.
- `--arena-segment-size TEXT` (default: `1GB`) segment size for arena provider.
- `-i, --interactive` run configured transforms, then prompt for extra transforms.
- `-s, --summarize / --no-summarize` (default: summarize) emit executed transform summary.
- `--summarize-path PATH` write summary to file (otherwise prints YAML to stdout).
- `--log-level TEXT` one of `debug`, `info`, `warning`, `error`, `critical`.
- `--install-completion` install shell completion for current shell.
- `--show-completion` print completion script for current shell.

## Plan format
```yaml
inputs:
  - model::/path/to/input.safetensors   # alias::path
  # with a single input, alias is optional and defaults to "model"

transforms:
  - dump: { target: ".*", format: compact }
  - copy: { from: ln_f.weight, to: ln_f_copy.weight }

output:
  path: /path/to/output                  # file or directory
  format: safetensors                    # optional: safetensors|torch
  shard: 5GB                             # optional (safetensors only)
```

Notes:
- `output` is optional. If omitted, no final checkpoint is written.
- Transforms are always a list of single-key mappings.

## Tensor references
Most transforms use tensor references:
- `alias::expr`
- `expr` (uses default model alias when available)
- `alias::expr::[slice]`

`expr` supports:
- Regex string matching (full-match).
- Structured path patterns (list form), matched against dot-separated tensor names.

Slices follow Python-like syntax in brackets, e.g. `[:8]`, `[:, :10]`, `[1:3, :]`.

## Structured expressions
Structured expressions are list tokens (instead of regex strings), for example:

```yaml
from: ["block", "$i", "weight"]
to:   ["backup", "${i}", "weight"]
```

Supported source tokens:
- `literal` exact segment
- `$x` capture one segment
- `*xs` capture zero or more segments
- `~REGEX` regex on one segment
- `~x::REGEX` bind one capture
- `~x,y::REGEX` bind multiple captures

Output tokens:
- literals (with `${x}` interpolation)
- `*xs` variadic splice

## Batch vs interactive mode
Batch mode (default):
- Executes configured transforms in order.
- Any transform failure aborts execution (raises error).

Interactive mode (`-i`):
- First executes configured transforms in batch mode.
- Then opens prompt for additional YAML transform blocks.
- On failure in an interactive block, logs error, stops only that submitted block, and returns to prompt.
- `exit` transform cleanly stops the execution loop.
- Interactive prompt uses a richer UI and supports tab completion for command names, payload keys, model aliases, and loaded tensor names.

Interactive prompt accepts:
- a single transform mapping, or
- a YAML list of transform mappings.

Special transforms:
- `help`
- `exit`

History is stored in `~/.brainsurgery_history`.

## Commands (transforms)
All registered transforms:

- `help`: show command/assert help (`help`, `help: copy`, `help: assert`, `help: { assert: equal }`).
- `exit`: stop current execution loop.
- `dump`: print tensor summaries (`format`: `json|tree|compact`, `verbosity`: `shape|stat|full`).
- `assert`: evaluate assertion expressions; does not modify tensors.
- `load`: load full state_dict (`path`, optional `alias`) or single tensor (`path` + `to`, optional `format`).
- `save`: save full state_dict (`path`, optional `alias`, `format`, `shard`) or single tensor (`target`).
- `copy`: clone source tensor(s) to new destination(s) (destination must not exist).
- `move`: move tensor slot(s) without copy (no slicing; destination must not exist).
- `delete`: delete tensor(s) by target pattern.
- `assign`: copy values into existing destination tensor(s), shape/dtype/device must match.
- `add`: `to = from_a + from_b` into existing destination(s).
- `subtract`: `to = from_a - from_b` into existing destination(s).
- `multiply`: `to = from_a * from_b` into existing destination(s).
- `matmul`: `to = from_a @ from_b` into new destination(s).
- `add_`: in-place `to += from`.
- `subtract_`: in-place `to -= from`.
- `scale`: create new tensor(s): `to = from * by`.
- `scale_`: in-place scaling of target(s).
- `cast`: create new tensor(s) with new dtype.
- `cast_`: in-place dtype cast.
- `clamp`: create new tensor(s) clamped by `min`/`max`.
- `clamp_`: in-place clamp.
- `reshape`: create new reshaped tensor(s) (`shape` allows one `-1`).
- `reshape_`: in-place reshape/rebind.
- `permute`: create new tensor(s) with reordered dimensions (`order` permutation).
- `fill`: create new tensor(s) using source shape:
  - `mode=constant` + `value`
  - `mode=rand` (+ `distribution`, `low/high` or `mean/std`, optional `seed`)
  - `mode=tensor` + `values` (broadcast allowed)
- `fill_`: in-place fill with same modes as `fill`.
- `split`: split one source tensor into multiple new tensors (`to` list, `sizes`, optional `dim`).
- `concat`: concatenate multiple source refs into one new tensor (`from` list, optional `dim`).
- `phlora_`: in-place low-rank reconstruction (`rank`) on 2D tensors.
- `phlora`: split 2D target tensors into low-rank factors (`target_a`, `target_b`, `rank`);
  optional `delete_original` (default `true`), `require_missing_dest` (default `true`).

## Assert expressions
Used via:

```yaml
- assert: { equal: { left: a.weight, right: b.weight, eps: 1e-6 } }
```

Supported operators:
- `exists`: reference matches at least one tensor.
- `count`: exact number of matches (`of`, `is`).
- `shape`: tensor shape check (`of`, `is` list of ints).
- `dimensions`: rank check (`of`, `is` int).
- `dtype`: dtype check (`of`, `is` dtype string).
- `equal`: pairwise equality between mapped tensors (`left`, `right`, optional `eps`).
- `iszero`: all-zero check (`of`, optional `eps`).
- `all`: all nested assertions succeed.
- `any`: at least one nested assertion succeeds.
- `not`: nested assertion fails.

## Output behavior
- If `output` is omitted: no final checkpoint is saved.
- If `output` is a directory-like path (or no suffix), safetensors output defaults to sharded writing using `--shard-size` unless shard is disabled.
- `torch` output requires file suffix `.pt`, `.pth`, or `.bin`.
- `output.shard` and `save.shard` are safetensors-only.

## Summary output
When summarize is enabled (default), brainsurgery emits the exact transforms that actually ran:
- to stdout by default
- or to `--summarize-path`

Useful for reproducibility when interactive edits or early `exit` changed execution.
