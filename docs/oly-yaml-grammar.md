# OLY-Expressible YAML Grammar

This document defines the exact YAML *data shape* representable by `OLY/1` as implemented in `brainsurgery/cli/oly.py`.

The grammar is over the parsed YAML value model (mapping/list/scalar), not YAML presentation details (flow/block style, quoting style, anchors, tags, comments).

## 1. AST Grammar (EBNF-like)

```ebnf
document        = transform_map ;

transform_map   = "{" , transform_name , ":" , payload_or_null , "}" ;
payload_or_null = payload_map | null ;
payload_map     = map_ident_to_value ;

value           = scalar | list | map_ident_to_value ;
list            = "[" , [ value , { "," , value } ] , "]" ;

map_ident_to_value = "{"
                     , [ ident , ":" , value
                         , { "," , ident , ":" , value } ]
                     , "}" ;

scalar          = string | bool | null | int | float ;
```

## 2. Identifier Constraints

All mapping keys that must round-trip through OLY parser/emitter are restricted to:

```regex
[A-Za-z_][A-Za-z0-9_-]*
```

This applies to:
- top-level transform name
- payload keys
- keys in nested inline maps

## 3. Scalar Constraints

From OLY parsing rules:
- `bool`: only `true`/`false` spellings are produced by canonical OLY emission.
- `null`: canonical OLY emission uses `null`.
- `int`: must be representable by OLY integer token grammar:
  - `-?(0|[1-9][0-9]*)`
- `float`: must be representable by OLY float token grammar:
  - `-?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+(?:[eE][+-]?[0-9]+)|[0-9]+\.[0-9]*[eE][+-]?[0-9]+)`
- `string`: any string is representable (emitter quotes when needed).

Notes:
- Values like NaN/Infinity are YAML floats but are *not* preserved as floats through OLY round-trip; treat them as outside this grammar.
- YAML aliases/tags/custom scalar types are outside this grammar.

## 4. Structural Constraints

- Exactly one top-level transform key is allowed.
- Top-level value must be a mapping or null.
  - Null is emitted as empty payload shorthand (`transform:`) and parses back as `{}`.
- Lists preserve order.
- Mappings preserve insertion order.
- Duplicate keys are invalid in OLY parsing contexts.

## 5. Canonical Mapping Intent

Any value in this grammar can be emitted as OLY and parsed back to the same AST (with the null-payload normalization: top-level `null -> {}`).
