# Axon DSL (Draft v1)

Axon is an indentation-sensitive, expression-oriented DSL for model graphs.

Core style:

- no mandatory curly braces,
- `do`-style headers with indentation blocks,
- single-assignment bindings with `<-`,
- module-path call sugar via `op@path(...)`,
- namespaced ops via `ns::op(...)`,
- expression-level conditional via `cond ? a : b`,
- explicit return via `return ...`,
- mixed composition styles: nested call, forward pipe `|>`, and bind `>>=`.

## 1) Syntax Principles

1. Whitespace is structural (Python-like blocks).
2. Every bound name is assigned once in a scope.
3. Module hierarchy drives parameter-path inference.
4. Tensor programs should read like linear algebra pipelines.

## 2) Binding and Calls

- Binding: `x1 <- expr`
- Multi-bind: `q, k, v <- expr1, expr2, expr3`
- Module-scoped op: `linear@attn.c_attn(x)`
- Namespaced op: `cache::update(past, k, v)`
- Ternary: `use_cache ? a : b`
- Header:
  - `name :: T1 -> ?T2 -> (T3, ?T4)`
  - `name a b = do`
- Composition (equivalent intent):
  - `y <- g(f(x))`
  - `y <- x |> f |> g`
  - `y <- f(x) >>= \\z -> g(z)`

## 3) Scopes and Parameters

Paths define parameter ownership:

- `linear@mlp.c_fc(x)` implies parameters at `...mlp.c_fc.weight/bias`.
- Inside repeats/modules, lexical scope prefixes are appended automatically.

## 4) Symbols and Types

Top-level constants define symbols used by module signatures and args:
`D = 768`, `L = 12`, `EPS = 1e-05`.

## 5) Core Forms

## 5.1 Tensor forms

- `embed@path(ids)`
- `linear@path(x)`
- `layernorm@path(x, eps=...)`
- `rmsnorm@path(x, eps=..., cast_float=..., unit_offset=...)`
- `act::gelu_new(x)` (and other activations)
- `attention(q, k, v, causal=true, mask=?, scale=?)`
- `reshape_heads(x, heads=H, head_dim=Hd)`
- `merge_heads(x)`
- `topk(x, k=K)`

## 5.2 Control/graph forms

- `x <- expr`
- `return expr`
- `for@path i <- [0..N) do` blocks
- ternary `c ? t : f`

## 5.3 KV-cache helpers

- `cache::update(past, k, v)`
- `cache::seq_len(past)`
- `cache::coalesce(k_all, v_all, k, v)`

KV remains optional at spec level; codegen/runtime may use hints to optimize decoding.

## 5.4 MoE helpers (composable)

- `router(x)`
- `topk(scores, k=K)`
- `select_tokens(hidden, topk_scores, topk_idx, expert=e)`
- `expert_ffn@experts(e, x, activation=...)`
- `scatter_add(accum, token_idx, updates, weights)`

## 6) Parallelism Hints

Hints are annotations, not semantics changes:

```axon
@tp(axis="hidden", parts=2)
@pp(stage=1)
@cp(axis="seq", parts=2)
h_attn :: Tensor[B,S,D] -> Tensor[B,S,D]
h_attn x = do
```

These are preserved for lowering/planning (TP/PP/CP), ignored by pure interpretation.

## 7) Example (Refined GPT-2 Block)

```axon
gpt2_block :: Tensor[B,S,D] -> ?Cache -> ?Bool -> (Tensor[B,S,D], ?Cache)
gpt2_block x past_kv use_cache = do
  x1 <- layernorm@ln_1(x, eps=1e-5)
  qkv <- linear@attn.c_attn(x1)
  q_lin, k_lin, v_lin <- split_last(qkv)
  q, k, v <- reshape_heads(q_lin, heads=H), reshape_heads(k_lin, heads=H), reshape_heads(v_lin, heads=H)
  k_all, v_all, present_kv <- use_cache ? cache::update(past_kv, k, v) : k, v, null
  k_ctx, v_ctx <- cache::coalesce(k_all, v_all, k, v)
  a_heads <- attention(q, k_ctx, v_ctx, causal=true)
  x2 <- x + linear@attn.c_proj(merge_heads(a_heads))
  x3 <- layernorm@ln_2(x2, eps=1e-5)
  m1 <- linear@mlp.c_fc(x3)
  m2 <- act::gelu_new(m1)
  y <- x2 + linear@mlp.c_proj(m2)
  return y, present_kv
```

## 8) Lowering to Synapse

Lowering is mechanical:

- `<-` bindings map to graph node outputs,
- `op@path(...)` maps to Synapse op + inferred parameter path,
- `for@...` maps to Synapse `op: repeat`,
- ternary maps to conditional graph nodes / `when`-guarded assignments,
- annotations map to planner metadata.

Synapse YAML remains the canonical machine-readable format. Axon is the readable authoring/rendering format.
