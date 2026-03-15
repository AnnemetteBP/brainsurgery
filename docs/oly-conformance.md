# OLY/1 Conformance Matrix

## 1. Goals
- Validate grammar coverage.
- Validate mapping to/from YAML.
- Lock down edge-case behavior.

## 2. Acceptance Cases
Each case should validate:
- `parse_oly(input) -> AST`
- `emit_yaml(AST) -> expected_yaml`
- `parse_yaml(expected_yaml) -> AST`
- `emit_oly(AST) -> expected_oly_canonical`

| ID | OLY Input | Expected YAML (shape) | Notes |
|---|---|---|---|
| A1 | `exit:` | `exit: {}` | Empty payload shorthand accepted. |
| A2 | `help: assert: all` | `help: { assert: all }` | Nested key/value payload. |
| A3 | `copy: from: x, to: y` | `copy: { from: x, to: y }` | Basic binary mapping. |
| A4 | `copy: from: [layer, $i, attn], to: [layer, $i, attention]` | `copy: { from: ["layer", "$i", "attn"], to: ["layer", "$i", "attention"] }` | Structured shorthand. |
| A5 | `copy: from: [*prefix, $i, attn, *suffix], to: [*prefix, ${i}, attention, *suffix]` | same mapping with quoted YAML scalars | `$i` and `${i}` equivalent in output context. |
| A6 | `prefixes: mode: rename, from: work, to: merged` | `prefixes: { mode: rename, from: work, to: merged }` | Existing transform payload. |
| A7 | `assert: exists: model::h.0.attn.bias` | `assert: { exists: model::h.0.attn.bias }` | Scalar with `::`. |
| A8 | `copy: from: "a,b", to: "c:d"` | `copy: { from: "a,b", to: "c:d" }` | Delimiters inside quoted scalars. |

## 3. Rejection Cases
| ID | Input | Expected Error Category | Notes |
|---|---|---|---|
| R1 | `` (empty) | empty-input | No transform present. |
| R2 | `copy` | missing-colon | No payload delimiter. |
| R3 | `copy: from x, to: y` | missing-key-colon | Invalid kv pair. |
| R4 | `copy: from: [a, b` | unclosed-list | Must report location. |
| R5 | `copy: from: "abc` | unclosed-quote | Must report location. |
| R6 | `copy: from: a, from: b` | duplicate-key | Deterministic rejection. |
| R7 | `1copy: from: a, to: b` | invalid-identifier | Transform key rule. |

## 4. Canonicalization Rules
- Single spaces after `:` and `,`.
- No leading/trailing spaces in canonical output.
- Preserve list and mapping order.
- Prefer bare scalars when safe, otherwise quote.
- Canonical output for equivalent inputs must match exactly.

## 5. Property Tests
Add fuzz/property tests over generated ASTs within grammar limits:
- `AST -> OLY -> AST'` equality.
- `AST -> YAML -> AST'` equality.
- `OLY -> AST -> OLY` idempotence after canonicalization.
- `YAML -> AST -> YAML` idempotence for canonical emitter.

## 6. Implementation Gates
- Parser implementation complete only when A* and R* cases pass.
- Round-trip guarantees required before enabling OLY as default interactive syntax.
