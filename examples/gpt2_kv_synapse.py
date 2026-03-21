from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class GPT2KVSynapse(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int | float | bool] = {
            "T": 1024,
            "D": 768,
            "L": 12,
            "H": 12,
            "V": 520257,
        }
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "GPT2KVSynapse":
        return cls(state_dict=state_dict)

    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._state = dict(state_dict)

    def _param(self, path: str) -> torch.Tensor:
        return self._state[path]

    def _join_scope(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return f"{left}.{right}"

    def _scope_of(self, node_path: str) -> str:
        if "." not in node_path:
            return ""
        return node_path.rsplit(".", 1)[0]

    def _safe_get(self, env: dict[str, Any], name: str) -> Any:
        if name not in env:
            raise ValueError(f"Missing variable in graph env: {name}")
        return env[name]

    def _prepare_env(
        self, input_ids: torch.Tensor | None, inputs: dict[str, Any], input_specs: dict[str, Any]
    ) -> dict[str, Any]:
        env = {"input_ids": input_ids, **inputs} if input_ids is not None else dict(inputs)
        for input_name, input_spec in input_specs.items():
            optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
            if input_name in env:
                continue
            if optional:
                env[input_name] = None
            else:
                raise ValueError(f"Missing required input: {input_name}")
        return env

    def _for_values(self, *, from_value: int, to_value: int, step_value: int):
        if not isinstance(from_value, int):
            raise ValueError(f"for _from must resolve to int, got {from_value!r}")
        if not isinstance(to_value, int):
            raise ValueError(f"for _to must resolve to int, got {to_value!r}")
        if not isinstance(step_value, int):
            raise ValueError(f"for _step must resolve to int, got {step_value!r}")
        if step_value == 0:
            raise ValueError("for _step must be non-zero")
        return range(from_value, to_value, step_value)

    def _block_Cache_past_length(self, cache, scope: str) -> tuple[Any, ...]:
        env: dict[str, Any] = {}
        env["cache"] = cache
        if cache is None:
            first_1 = None
        else:
            try:
                first_1 = cache[int(0)]
            except (IndexError, KeyError, TypeError):
                first_1 = None
        if first_1 is None:
            out_0_2 = 0
        else:
            out_0_2 = int(first_1[0].shape[-2])
        return out_0_2

    def _block_gpt2_block(self, x, attn_mask, past_kv, scope: str) -> tuple[Any, ...]:
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        env["attn_mask"] = attn_mask
        env["past_kv"] = past_kv
        node_path_3 = self._join_scope(scope, "n_op_1")
        x1_4 = F.layer_norm(
            x,
            (x.shape[-1],),
            weight=emitter._param(self._join_scope(self._scope_of(node_path_3), "ln_1.weight")),
            bias=emitter._param(self._join_scope(self._scope_of(node_path_3), "ln_1.bias")),
            eps=float(1e-05),
        )
        node_path_5 = self._join_scope(scope, "n_call_3")
        if x1_4.numel() == 0:
            pipe_2_6 = x1_4.new_empty(
                (
                    *x1_4.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_5), "attn.c_attn.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            pipe_2_6 = torch.matmul(
                x1_4,
                self._param(self._join_scope(self._scope_of(node_path_5), "attn.c_attn.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_5), "attn.c_attn.bias"))
                is not None
            ):
                pipe_2_6 = pipe_2_6 + self._state.get(
                    self._join_scope(self._scope_of(node_path_5), "attn.c_attn.bias")
                )
        split_7 = torch.chunk(pipe_2_6, int(3), dim=-1)
        q_lin_8 = split_7[0]
        k_lin_9 = split_7[1]
        v_lin_10 = split_7[2]
        heads_11 = int(12)
        head_dim_12 = None
        if heads_11 is None and head_dim_12 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_11 is None:
            if q_lin_8.shape[-1] % int(head_dim_12) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_11 = q_lin_8.shape[-1] // int(head_dim_12)
        if head_dim_12 is None:
            if q_lin_8.shape[-1] % int(heads_11) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_12 = q_lin_8.shape[-1] // int(heads_11)
        expected_hidden_13 = int(heads_11) * int(head_dim_12)
        if q_lin_8.shape[-1] != expected_hidden_13:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        q_14 = q_lin_8.view(
            q_lin_8.shape[0], q_lin_8.shape[1], int(heads_11), int(head_dim_12)
        ).transpose(1, 2)
        heads_15 = int(12)
        head_dim_16 = None
        if heads_15 is None and head_dim_16 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_15 is None:
            if k_lin_9.shape[-1] % int(head_dim_16) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_15 = k_lin_9.shape[-1] // int(head_dim_16)
        if head_dim_16 is None:
            if k_lin_9.shape[-1] % int(heads_15) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_16 = k_lin_9.shape[-1] // int(heads_15)
        expected_hidden_17 = int(heads_15) * int(head_dim_16)
        if k_lin_9.shape[-1] != expected_hidden_17:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        k_18 = k_lin_9.view(
            k_lin_9.shape[0], k_lin_9.shape[1], int(heads_15), int(head_dim_16)
        ).transpose(1, 2)
        heads_19 = int(12)
        head_dim_20 = None
        if heads_19 is None and head_dim_20 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_19 is None:
            if v_lin_10.shape[-1] % int(head_dim_20) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_19 = v_lin_10.shape[-1] // int(head_dim_20)
        if head_dim_20 is None:
            if v_lin_10.shape[-1] % int(heads_19) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_20 = v_lin_10.shape[-1] // int(heads_19)
        expected_hidden_21 = int(heads_19) * int(head_dim_20)
        if v_lin_10.shape[-1] != expected_hidden_21:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        v_22 = v_lin_10.view(
            v_lin_10.shape[0], v_lin_10.shape[1], int(heads_19), int(head_dim_20)
        ).transpose(1, 2)
        if past_kv is None:
            k_18 = k_18
            v_22 = v_22
        else:
            k_18 = torch.cat([past_kv[0], k_18], dim=-2)
            v_22 = torch.cat([past_kv[1], v_22], dim=-2)
        new_kv_23 = (k_18, v_22)
        q_len_25 = q_14.shape[-2]
        k_len_26 = k_18.shape[-2]
        if not hasattr(self, "_causal_mask_cache"):
            self._causal_mask_cache = {}
        window_33 = int(1024)
        if attn_mask is None:
            pad_key_34 = None
        else:
            pad_key_34 = (
                int(attn_mask.data_ptr()),
                int(attn_mask.storage_offset()),
                tuple(int(x) for x in attn_mask.shape),
            )
        cache_key_27 = (
            int(q_len_25),
            int(k_len_26),
            window_33,
            q_14.dtype,
            q_14.device,
            pad_key_34,
        )
        cached_mask_28 = self._causal_mask_cache.get(cache_key_27)
        if torch.is_tensor(cached_mask_28):
            mask_24 = cached_mask_28
        else:
            j_idx_30 = torch.arange(k_len_26, device=q_14.device).unsqueeze(0)
            if q_len_25 == 1:
                keep_31 = j_idx_30 >= (k_len_26 - window_33)
            else:
                i_idx_29 = torch.arange(q_len_25, device=q_14.device).unsqueeze(1)
                keep_31 = j_idx_30 <= i_idx_29
                keep_31 = keep_31 & (j_idx_30 >= (i_idx_29 - window_33 + 1))
            if attn_mask is not None:
                if attn_mask.ndim != 2:
                    raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
                if int(attn_mask.shape[-1]) != k_len_26:
                    raise ValueError(
                        "causal_mask.padding_mask width must match key sequence length"
                    )
                pad_keep_35 = attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
                keep_31 = keep_31.unsqueeze(0).unsqueeze(0) & pad_keep_35
            else:
                keep_31 = keep_31.view(1, 1, q_len_25, k_len_26)
            mask_val_32 = torch.finfo(q_14.dtype).min
            mask_24 = torch.where(
                keep_31,
                torch.zeros((), dtype=q_14.dtype, device=q_14.device),
                torch.full((), mask_val_32, dtype=q_14.dtype, device=q_14.device),
            )
            self._causal_mask_cache[cache_key_27] = mask_24
        pipe_10_36 = F.scaled_dot_product_attention(
            q_14,
            k_18,
            v_22,
            attn_mask=mask_24,
            dropout_p=0.0,
            is_causal=(q_14.shape[-2] > 1 and mask_24 is None),
            scale=None,
        )
        pipe_12_37 = (
            pipe_10_36.transpose(1, 2)
            .contiguous()
            .view(
                pipe_10_36.shape[0], pipe_10_36.shape[2], pipe_10_36.shape[1] * pipe_10_36.shape[3]
            )
        )
        node_path_38 = self._join_scope(scope, "n_call_14")
        if pipe_12_37.numel() == 0:
            a_39 = pipe_12_37.new_empty(
                (
                    *pipe_12_37.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_38), "attn.c_proj.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            a_39 = torch.matmul(
                pipe_12_37,
                self._param(self._join_scope(self._scope_of(node_path_38), "attn.c_proj.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_38), "attn.c_proj.bias"))
                is not None
            ):
                a_39 = a_39 + self._state.get(
                    self._join_scope(self._scope_of(node_path_38), "attn.c_proj.bias")
                )
        x = x + a_39
        node_path_40 = self._join_scope(scope, "n_op_16")
        x3_41 = F.layer_norm(
            x,
            (x.shape[-1],),
            weight=emitter._param(self._join_scope(self._scope_of(node_path_40), "ln_2.weight")),
            bias=emitter._param(self._join_scope(self._scope_of(node_path_40), "ln_2.bias")),
            eps=float(1e-05),
        )
        node_path_42 = self._join_scope(scope, "n_call_18")
        if x3_41.numel() == 0:
            pipe_17_43 = x3_41.new_empty(
                (
                    *x3_41.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_42), "mlp.c_fc.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            pipe_17_43 = torch.matmul(
                x3_41,
                self._param(self._join_scope(self._scope_of(node_path_42), "mlp.c_fc.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_42), "mlp.c_fc.bias"))
                is not None
            ):
                pipe_17_43 = pipe_17_43 + self._state.get(
                    self._join_scope(self._scope_of(node_path_42), "mlp.c_fc.bias")
                )
        pipe_19_44 = (
            0.5
            * pipe_17_43
            * (
                1.0
                + torch.tanh(
                    0.7978845608028654
                    * (pipe_17_43 + 0.044715 * pipe_17_43 * pipe_17_43 * pipe_17_43)
                )
            )
        )
        node_path_45 = self._join_scope(scope, "n_call_21")
        if pipe_19_44.numel() == 0:
            out_0_46 = pipe_19_44.new_empty(
                (
                    *pipe_19_44.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_45), "mlp.c_proj.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            out_0_46 = torch.matmul(
                pipe_19_44,
                self._param(self._join_scope(self._scope_of(node_path_45), "mlp.c_proj.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_45), "mlp.c_proj.bias"))
                is not None
            ):
                out_0_46 = out_0_46 + self._state.get(
                    self._join_scope(self._scope_of(node_path_45), "mlp.c_proj.bias")
                )
        m_47 = out_0_46
        y_48 = x + m_47
        return (y_48, new_kv_23)

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        input_specs = {
            "input_ids": {"optional": False},
            "attn_mask": {"optional": True},
            "past_kv": {"optional": True},
            "use_cache": {"optional": True},
        }
        env = self._prepare_env(input_ids, inputs, input_specs)
        scope = ""
        emitter = self
        input_ids = self._safe_get(env, "input_ids")
        attn_mask = env.get("attn_mask")
        past_kv = env.get("past_kv")
        use_cache = env.get("use_cache")
        node_path_49 = self._join_scope(scope, "n_op_1")
        tok_50 = F.embedding(
            input_ids, emitter._param(self._join_scope(self._scope_of(node_path_49), "wte.weight"))
        )
        out_0_51 = self._block_Cache_past_length(cache=past_kv, scope=scope)
        kwarg_3_52 = out_0_51
        if input_ids.ndim != 2:
            raise ValueError("position_ids._args must resolve to rank-2 [batch, seq] tensor")
        if attn_mask is not None:
            if attn_mask.ndim != 2:
                raise ValueError("position_ids.attention_mask must be rank-2 [batch, seq]")
            if int(attn_mask.shape[0]) != int(input_ids.shape[0]):
                raise ValueError("position_ids.attention_mask batch size must match input")
            if int(attn_mask.shape[1]) < int(input_ids.shape[1]):
                raise ValueError(
                    "position_ids.attention_mask width must be >= input sequence length"
                )
            full_pos_55 = attn_mask.to(torch.long).cumsum(dim=-1) - 1
            full_pos_55 = full_pos_55.masked_fill(attn_mask == 0, 0)
            pipe_2_53 = full_pos_55[:, -input_ids.shape[1] :]
        else:
            pos_offset_54 = int(kwarg_3_52)
            if pos_offset_54 < 0:
                raise ValueError("position_ids.past_length must resolve to non-negative int")
            pipe_2_53 = torch.arange(
                pos_offset_54,
                pos_offset_54 + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
        node_path_56 = self._join_scope(scope, "n_op_6")
        pos_57 = F.embedding(
            pipe_2_53, emitter._param(self._join_scope(self._scope_of(node_path_56), "wpe.weight"))
        )
        x_58 = tok_50 + pos_57
        new_kv_59 = None
        if use_cache:
            new_kv_59 = []
        if not (use_cache):
            new_kv_59 = None
        to_61 = int(12)
        from_62 = int(0)
        step_63 = int(1)
        for i_60 in self._for_values(from_value=from_62, to_value=to_61, step_value=step_63):
            scope_64 = self._join_scope(scope, f"h.{i_60}")
            if past_kv is None:
                past_i_65 = None
            else:
                try:
                    past_i_65 = past_kv[int(i_60)]
                except (IndexError, KeyError, TypeError):
                    past_i_65 = None
            y_66, new_kv_67 = self._block_gpt2_block(
                x=x_58, attn_mask=attn_mask, past_kv=past_i_65, scope=scope_64
            )
            x_58 = y_66
            new_i_68 = new_kv_67
            if new_kv_59 is None:
                new_kv_59 = None
            else:
                new_kv_59 = list(new_kv_59)
                new_kv_59.append(new_i_68)
        node_path_69 = self._join_scope(scope, "n_op_15")
        pipe_14_70 = F.layer_norm(
            x_58,
            (x_58.shape[-1],),
            weight=emitter._param(self._join_scope(self._scope_of(node_path_69), "ln_f.weight")),
            bias=emitter._param(self._join_scope(self._scope_of(node_path_69), "ln_f.bias")),
            eps=float(1e-05),
        )
        node_path_71 = self._join_scope(scope, "n_op_16")
        if pipe_14_70.numel() == 0:
            logits_72 = pipe_14_70.new_empty(
                (
                    *pipe_14_70.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_71), "wte.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            logits_72 = F.linear(
                pipe_14_70,
                self._param(self._join_scope(self._scope_of(node_path_71), "wte.weight")),
                None,
            )
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_72
        outputs["new_kv"] = new_kv_59
        if "logits" in outputs and len(outputs) == 1:
            return outputs["logits"]
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        eos_token_id: int,
        max_len: int,
        attention_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if attention_mask is not None and attn_mask is not None:
            raise ValueError("pass at most one of attention_mask or attn_mask")
        mask = attention_mask if attention_mask is not None else attn_mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("attention_mask must be rank-2 [batch, seq]")
            if mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have same shape as input_ids")
        if input_ids.size(1) >= max_len:
            return input_ids[:, :max_len]

        batch, start_len = input_ids.shape
        generated = input_ids.new_empty((batch, max_len))
        generated[:, :start_len] = input_ids
        generated_mask = None
        if mask is not None:
            generated_mask = mask.new_zeros((batch, max_len))
            generated_mask[:, :start_len] = mask
        cache_state = None
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        cur_len = start_len
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                while cur_len < max_len and not torch.all(finished):
                    step_input = (
                        generated[:, :cur_len]
                        if cache_state is None
                        else generated[:, cur_len - 1 : cur_len]
                    )
                    call_kwargs: dict[str, Any] = {}
                    if generated_mask is not None:
                        call_kwargs["attention_mask"] = generated_mask[:, :cur_len]
                        call_kwargs["attn_mask"] = generated_mask[:, :cur_len]
                    call_kwargs["past_kv"] = cache_state
                    call_kwargs["use_cache"] = True
                    model_out = self.forward(step_input, **call_kwargs)
                    if isinstance(model_out, dict):
                        logits = model_out["logits"]
                        if "new_kv" in model_out:
                            cache_state = model_out["new_kv"]
                    else:
                        logits = model_out
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    next_token = torch.where(
                        finished, torch.full_like(next_token, eos_token_id), next_token
                    )
                    generated[:, cur_len] = next_token
                    finished = torch.logical_or(finished, next_token == eos_token_id)
                    if generated_mask is not None:
                        generated_mask[:, cur_len] = 1
                    cur_len += 1
        finally:
            if was_training:
                self.train()
        return generated[:, :cur_len]
