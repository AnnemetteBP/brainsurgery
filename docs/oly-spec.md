# OLY/1 Specification (One Line YAML)

## 1. Purpose
OLY/1 is a single-line command syntax for interactive transforms that maps directly to the existing transform YAML shape.

Example:

`copy: from: [*prefix, $i, attn, *suffix], to: [*prefix, $i, attention, *suffix]`

maps to:

`copy: { from: ["*prefix", "$i", "attn", "*suffix"], to: ["*prefix", "$i", "attention", "*suffix"] }`

## 2. Scope
- Exactly one transform per line.
- One top-level transform name.
- Payload is a comma-separated list of key/value pairs.
- Values support scalar, list, and inline map.
- No comments in OLY/1.

## 3. Top-Level Shape
OLY/1 line:

`<transform_name>`
or
`<transform_name> : <payload_kv>`

Equivalent YAML shape:

`{ <transform_name>: { <payload_kv_as_mapping> } }`

The bare single-word shorthand `<transform_name>` is equivalent to an empty payload mapping.

## 4. Lexical Rules
- Whitespace: space and tab are insignificant between tokens unless inside quoted strings.
- Identifiers:
  - `IDENT_START = [A-Za-z_]`
  - `IDENT_REST  = [A-Za-z0-9_-]`
  - `IDENT       = IDENT_START IDENT_REST*`
- Quoted strings:
  - Single quotes: `'...'`
  - Double quotes: `"..."`
  - Backslash escaping is only recognized in double-quoted strings.
- Bare scalars:
  - Allowed when not containing delimiters `, : [ ] { } #` or whitespace.
  - Tokens beginning with YAML-special characters that would otherwise be ambiguous SHOULD be quoted in canonical YAML output.

## 5. Grammar (EBNF)
```ebnf
line            = ws, transform_name, ws
                | ws, transform_name, ws, ":", ws, payload, ws ;
payload         = payload_kv | inline_map ;
payload_kv      = kv_pair, { ws, ",", ws, kv_pair } ;
kv_pair         = key, ws, ":", ws, value ;

value           = scalar | list | inline_map ;
list            = "[", ws, [ value, { ws, ",", ws, value } ], ws, "]" ;
inline_map      = "{", ws, [ kv_pair, { ws, ",", ws, kv_pair } ], ws, "}" ;

scalar          = quoted | bare ;
quoted          = sq_string | dq_string ;
sq_string       = "'", { sq_char }, "'" ;
dq_string       = "\"", { dq_char }, "\"" ;
bare            = bare_char, { bare_char } ;

transform_name  = ident ;
key             = ident ;
ident           = ident_start, { ident_rest } ;
ident_start     = "A"..."Z" | "a"..."z" | "_" ;
ident_rest      = ident_start | "0"..."9" | "-" ;

ws              = { " " | "\t" } ;
```

## 6. Structured Path Segment Semantics
When list values are used for structured path mapping (for example `from`/`to` in `copy`, `move`, `assign`, `add`, etc.), segments are interpreted as:

- `$name`: scalar capture/interpolation token.
- `${name}`: explicit interpolation form, equivalent to `$name`.
- `*name`: variadic segment token.
- `~...`: structured regex segment token (source side only where supported by matcher rules).
- Any other token: literal segment.

Literal segments that intentionally begin with `$`, `*`, or `~` should be quoted.

## 7. Mapping to YAML
The parser MUST produce an AST that can be emitted to canonical YAML:

- Top-level: one-key mapping `{transform: payload_map}`.
- Payload: mapping with deterministic key order (input order preserved unless canonicalizer is configured otherwise).
- Lists: preserved in order.
- Scalars:
  - Emit as plain scalars when safe.
  - Emit as quoted scalars when required for correctness (especially values like `*prefix` and `${i}`).

## 8. Round-Trip Invariants
For any valid OLY/1 line `L`:

1. Parse `L -> AST`.
2. Emit canonical YAML `AST -> Y`.
3. Parse YAML `Y -> AST2`.
4. Emit canonical OLY `AST2 -> L2`.

Required:
- `AST == AST2`.
- `canonicalize(L) == L2`.

## 9. Error Behavior
Parser errors MUST include:
- A short message.
- Character offset or token context.
- Suggested fix when possible.

Examples:
- Duplicate key in payload.
- Unexpected token after value.
- Unclosed quote/list/map.
- Invalid identifier for transform/key.

## 10. Compatibility and Versioning
- This spec is `OLY/1`.
- Future revisions must preserve `OLY/1` behavior under explicit version selection or auto-detection fallback.

## 11. YAML Subset Reference
For the exact YAML value grammar representable by OLY, see:

- [OLY-Expressible YAML Grammar](/Users/petersk/Nobackup/brainsurgery/docs/oly-yaml-grammar.md)
