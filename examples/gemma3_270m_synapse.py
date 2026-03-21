from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Gemma3Synapse270M(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int | float | bool] = {
            "D": 640,
            "V": 262208,
            "L": 18,
            "H": 4,
            "KVH": 1,
            "QD": 1024,
            "KVD": 256,
            "FFN": 2048,
            "ROPE_PERIOD": 6,
            "THETA_BASE": 10000.0,
            "THETA_LONG": 1000000.0,
            "WIN_LOCAL": 512,
            "WIN_LONG": 32768,
            "ATTN_SCALE": 0.0625,
            "EMB_SCALE": 25.298221281347036,
        }
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "Gemma3Synapse270M":
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

    def _block_gemma3_block(self, x, i, pos_ids, attn_mask, past_kv, scope: str) -> tuple[Any, ...]:
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        env["i"] = i
        env["pos_ids"] = pos_ids
        env["attn_mask"] = attn_mask
        env["past_kv"] = past_kv
        node_path_3 = self._join_scope(scope, "n_call_1")
        xnorm_5 = x.float() * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        x_norm_4 = xnorm_5 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_3), "input_layernorm.weight")
            ).float()
        )
        x_norm_4 = x_norm_4.type_as(x)
        node_path_6 = self._join_scope(scope, "n_op_3")
        if x_norm_4.numel() == 0:
            pipe_2_7 = x_norm_4.new_empty(
                (
                    *x_norm_4.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_6), "self_attn.q_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_2_7 = F.linear(
                x_norm_4,
                self._param(
                    self._join_scope(self._scope_of(node_path_6), "self_attn.q_proj.weight")
                ),
                None,
            )
        heads_8 = int(4)
        head_dim_9 = None
        if heads_8 is None and head_dim_9 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_8 is None:
            if pipe_2_7.shape[-1] % int(head_dim_9) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_8 = pipe_2_7.shape[-1] // int(head_dim_9)
        if head_dim_9 is None:
            if pipe_2_7.shape[-1] % int(heads_8) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_9 = pipe_2_7.shape[-1] // int(heads_8)
        expected_hidden_10 = int(heads_8) * int(head_dim_9)
        if pipe_2_7.shape[-1] != expected_hidden_10:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        pipe_4_11 = pipe_2_7.view(
            pipe_2_7.shape[0], pipe_2_7.shape[1], int(heads_8), int(head_dim_9)
        ).transpose(1, 2)
        node_path_12 = self._join_scope(scope, "n_call_6")
        xnorm_14 = pipe_4_11.float() * torch.rsqrt(
            torch.mean(pipe_4_11.float() * pipe_4_11.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        q_13 = xnorm_14 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_12), "self_attn.q_norm.weight")
            ).float()
        )
        q_13 = q_13.type_as(pipe_4_11)
        node_path_15 = self._join_scope(scope, "n_op_8")
        if x_norm_4.numel() == 0:
            pipe_7_16 = x_norm_4.new_empty(
                (
                    *x_norm_4.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_15), "self_attn.k_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_7_16 = F.linear(
                x_norm_4,
                self._param(
                    self._join_scope(self._scope_of(node_path_15), "self_attn.k_proj.weight")
                ),
                None,
            )
        heads_17 = int(1)
        head_dim_18 = None
        if heads_17 is None and head_dim_18 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_17 is None:
            if pipe_7_16.shape[-1] % int(head_dim_18) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_17 = pipe_7_16.shape[-1] // int(head_dim_18)
        if head_dim_18 is None:
            if pipe_7_16.shape[-1] % int(heads_17) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_18 = pipe_7_16.shape[-1] // int(heads_17)
        expected_hidden_19 = int(heads_17) * int(head_dim_18)
        if pipe_7_16.shape[-1] != expected_hidden_19:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        pipe_9_20 = pipe_7_16.view(
            pipe_7_16.shape[0], pipe_7_16.shape[1], int(heads_17), int(head_dim_18)
        ).transpose(1, 2)
        node_path_21 = self._join_scope(scope, "n_call_11")
        xnorm_23 = pipe_9_20.float() * torch.rsqrt(
            torch.mean(pipe_9_20.float() * pipe_9_20.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        k_22 = xnorm_23 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_21), "self_attn.k_norm.weight")
            ).float()
        )
        k_22 = k_22.type_as(pipe_9_20)
        node_path_24 = self._join_scope(scope, "n_op_13")
        if x_norm_4.numel() == 0:
            pipe_12_25 = x_norm_4.new_empty(
                (
                    *x_norm_4.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_24), "self_attn.v_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_12_25 = F.linear(
                x_norm_4,
                self._param(
                    self._join_scope(self._scope_of(node_path_24), "self_attn.v_proj.weight")
                ),
                None,
            )
        heads_26 = int(1)
        head_dim_27 = None
        if heads_26 is None and head_dim_27 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_26 is None:
            if pipe_12_25.shape[-1] % int(head_dim_27) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_26 = pipe_12_25.shape[-1] // int(head_dim_27)
        if head_dim_27 is None:
            if pipe_12_25.shape[-1] % int(heads_26) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_27 = pipe_12_25.shape[-1] // int(heads_26)
        expected_hidden_28 = int(heads_26) * int(head_dim_27)
        if pipe_12_25.shape[-1] != expected_hidden_28:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        v_29 = pipe_12_25.view(
            pipe_12_25.shape[0], pipe_12_25.shape[1], int(heads_26), int(head_dim_27)
        ).transpose(1, 2)
        half_30 = q_13.shape[-1] // 2
        inv_freq_31 = 1.0 / (
            float(10000.0 + (1000000.0 - 10000.0) * (((i + 1) % 6) == 0))
            ** (torch.arange(0, half_30, device=q_13.device, dtype=q_13.dtype) / float(half_30))
        )
        if pos_ids is not None:
            if pos_ids.ndim != 2:
                raise ValueError("apply_rope_pair.position_ids must be rank-2 [batch, seq]")
            if int(pos_ids.shape[0]) != int(q_13.shape[0]):
                raise ValueError("apply_rope_pair.position_ids batch size must match q/k batch")
            if int(pos_ids.shape[1]) != int(q_13.shape[-2]):
                raise ValueError(
                    "apply_rope_pair.position_ids width must match q/k sequence length"
                )
            pos_32 = pos_ids.to(device=q_13.device, dtype=q_13.dtype)
            ang_33 = pos_32.unsqueeze(-1) * inv_freq_31.unsqueeze(0).unsqueeze(0)
            cos_34 = torch.cos(ang_33).unsqueeze(1)
            sin_35 = torch.sin(ang_33).unsqueeze(1)
        else:
            pos_32 = torch.arange(
                int(0), int(0) + q_13.shape[-2], device=q_13.device, dtype=q_13.dtype
            )
            ang_33 = torch.einsum("t,d->td", pos_32, inv_freq_31)
            cos_34 = torch.cos(ang_33)[None, None, :, :]
            sin_35 = torch.sin(ang_33)[None, None, :, :]
        q1_36 = q_13[..., :half_30]
        q2_37 = q_13[..., half_30 : 2 * half_30]
        k1_38 = k_22[..., :half_30]
        k2_39 = k_22[..., half_30 : 2 * half_30]
        q_13 = torch.cat([q1_36 * cos_34 - q2_37 * sin_35, q1_36 * sin_35 + q2_37 * cos_34], dim=-1)
        k_22 = torch.cat([k1_38 * cos_34 - k2_39 * sin_35, k1_38 * sin_35 + k2_39 * cos_34], dim=-1)
        if past_kv is None:
            k_22 = k_22
            v_29 = v_29
        else:
            k_22 = torch.cat([past_kv[0], k_22], dim=-2)
            v_29 = torch.cat([past_kv[1], v_29], dim=-2)
        new_kv_40 = (k_22, v_29)
        n_rep_41 = int(4.0)
        if n_rep_41 == 1:
            k_22 = k_22
        else:
            k_22 = (
                k_22[:, :, None, :, :]
                .expand(k_22.shape[0], k_22.shape[1], n_rep_41, k_22.shape[2], k_22.shape[3])
                .reshape(k_22.shape[0], k_22.shape[1] * n_rep_41, k_22.shape[2], k_22.shape[3])
            )
        n_rep_42 = int(4.0)
        if n_rep_42 == 1:
            v_29 = v_29
        else:
            v_29 = (
                v_29[:, :, None, :, :]
                .expand(v_29.shape[0], v_29.shape[1], n_rep_42, v_29.shape[2], v_29.shape[3])
                .reshape(v_29.shape[0], v_29.shape[1] * n_rep_42, v_29.shape[2], v_29.shape[3])
            )
        q_len_44 = q_13.shape[-2]
        k_len_45 = k_22.shape[-2]
        if not hasattr(self, "_causal_mask_cache"):
            self._causal_mask_cache = {}
        window_52 = int(512 + (32768 - 512) * (((i + 1) % 6) == 0))
        if attn_mask is None:
            pad_key_53 = None
        else:
            pad_key_53 = (
                int(attn_mask.data_ptr()),
                int(attn_mask.storage_offset()),
                tuple(int(x) for x in attn_mask.shape),
            )
        cache_key_46 = (
            int(q_len_44),
            int(k_len_45),
            window_52,
            q_13.dtype,
            q_13.device,
            pad_key_53,
        )
        cached_mask_47 = self._causal_mask_cache.get(cache_key_46)
        if torch.is_tensor(cached_mask_47):
            mask_43 = cached_mask_47
        else:
            j_idx_49 = torch.arange(k_len_45, device=q_13.device).unsqueeze(0)
            if q_len_44 == 1:
                keep_50 = j_idx_49 >= (k_len_45 - window_52)
            else:
                i_idx_48 = torch.arange(q_len_44, device=q_13.device).unsqueeze(1)
                keep_50 = j_idx_49 <= i_idx_48
                keep_50 = keep_50 & (j_idx_49 >= (i_idx_48 - window_52 + 1))
            if attn_mask is not None:
                if attn_mask.ndim != 2:
                    raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
                if int(attn_mask.shape[-1]) != k_len_45:
                    raise ValueError(
                        "causal_mask.padding_mask width must match key sequence length"
                    )
                pad_keep_54 = attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
                keep_50 = keep_50.unsqueeze(0).unsqueeze(0) & pad_keep_54
            else:
                keep_50 = keep_50.view(1, 1, q_len_44, k_len_45)
            mask_val_51 = torch.finfo(q_13.dtype).min
            mask_43 = torch.where(
                keep_50,
                torch.zeros((), dtype=q_13.dtype, device=q_13.device),
                torch.full((), mask_val_51, dtype=q_13.dtype, device=q_13.device),
            )
            self._causal_mask_cache[cache_key_46] = mask_43
        pipe_20_55 = F.scaled_dot_product_attention(
            q_13,
            k_22,
            v_29,
            attn_mask=mask_43,
            dropout_p=0.0,
            is_causal=(q_13.shape[-2] > 1 and mask_43 is None),
            scale=0.0625,
        )
        pipe_22_56 = (
            pipe_20_55.transpose(1, 2)
            .contiguous()
            .view(
                pipe_20_55.shape[0], pipe_20_55.shape[2], pipe_20_55.shape[1] * pipe_20_55.shape[3]
            )
        )
        node_path_57 = self._join_scope(scope, "n_op_24")
        if pipe_22_56.numel() == 0:
            attn_58 = pipe_22_56.new_empty(
                (
                    *pipe_22_56.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_57), "self_attn.o_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            attn_58 = F.linear(
                pipe_22_56,
                self._param(
                    self._join_scope(self._scope_of(node_path_57), "self_attn.o_proj.weight")
                ),
                None,
            )
        node_path_59 = self._join_scope(scope, "n_call_25")
        xnorm_60 = attn_58.float() * torch.rsqrt(
            torch.mean(attn_58.float() * attn_58.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        attn_58 = xnorm_60 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_59), "post_attention_layernorm.weight")
            ).float()
        )
        attn_58 = attn_58.type_as(attn_58)
        x2_61 = x + attn_58
        node_path_62 = self._join_scope(scope, "n_call_27")
        xnorm_64 = x2_61.float() * torch.rsqrt(
            torch.mean(x2_61.float() * x2_61.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        x3_63 = xnorm_64 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_62), "pre_feedforward_layernorm.weight")
            ).float()
        )
        x3_63 = x3_63.type_as(x2_61)
        node_path_65 = self._join_scope(scope, "n_op_28")
        if x3_63.numel() == 0:
            g_66 = x3_63.new_empty(
                (
                    *x3_63.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_65), "mlp.gate_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            g_66 = F.linear(
                x3_63,
                self._param(self._join_scope(self._scope_of(node_path_65), "mlp.gate_proj.weight")),
                None,
            )
        node_path_67 = self._join_scope(scope, "n_op_29")
        if x3_63.numel() == 0:
            u_68 = x3_63.new_empty(
                (
                    *x3_63.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_67), "mlp.up_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            u_68 = F.linear(
                x3_63,
                self._param(self._join_scope(self._scope_of(node_path_67), "mlp.up_proj.weight")),
                None,
            )
        pipe_30_69 = (
            0.5
            * g_66
            * (1.0 + torch.tanh(0.7978845608028654 * (g_66 + 0.044715 * g_66 * g_66 * g_66)))
        )
        pipe_32_70 = pipe_30_69 * u_68
        node_path_71 = self._join_scope(scope, "n_op_34")
        if pipe_32_70.numel() == 0:
            out_0_72 = pipe_32_70.new_empty(
                (
                    *pipe_32_70.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_71), "mlp.down_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            out_0_72 = F.linear(
                pipe_32_70,
                self._param(self._join_scope(self._scope_of(node_path_71), "mlp.down_proj.weight")),
                None,
            )
        m_73 = out_0_72
        node_path_74 = self._join_scope(scope, "n_call_36")
        xnorm_75 = m_73.float() * torch.rsqrt(
            torch.mean(m_73.float() * m_73.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        m_73 = xnorm_75 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_74), "post_feedforward_layernorm.weight")
            ).float()
        )
        m_73 = m_73.type_as(m_73)
        y_76 = x2_61 + m_73
        return (y_76, new_kv_40)

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
        out_0_77 = self._block_Cache_past_length(cache=past_kv, scope=scope)
        kwarg_1_78 = out_0_77
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
            full_pos_81 = attn_mask.to(torch.long).cumsum(dim=-1) - 1
            full_pos_81 = full_pos_81.masked_fill(attn_mask == 0, 0)
            pos_ids_79 = full_pos_81[:, -input_ids.shape[1] :]
        else:
            pos_offset_80 = int(kwarg_1_78)
            if pos_offset_80 < 0:
                raise ValueError("position_ids.past_length must resolve to non-negative int")
            pos_ids_79 = torch.arange(
                pos_offset_80,
                pos_offset_80 + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
        node_path_82 = self._join_scope(scope, "n_op_4")
        x_83 = F.embedding(
            input_ids,
            emitter._param(
                self._join_scope(self._scope_of(node_path_82), "model.embed_tokens.weight")
            ),
        )
        x_83 = x_83 * torch.tensor(float(25.298221281347036), dtype=x_83.dtype, device=x_83.device)
        new_kv_84 = None
        if use_cache:
            new_kv_84 = []
        if not (use_cache):
            new_kv_84 = None
        to_86 = int(18)
        from_87 = int(0)
        step_88 = int(1)
        for i_85 in self._for_values(from_value=from_87, to_value=to_86, step_value=step_88):
            scope_89 = self._join_scope(scope, f"model.layers.{i_85}")
            if past_kv is None:
                past_i_90 = None
            else:
                try:
                    past_i_90 = past_kv[int(i_85)]
                except (IndexError, KeyError, TypeError):
                    past_i_90 = None
            y_91, new_kv_92 = self._block_gemma3_block(
                x=x_83,
                i=i_85,
                pos_ids=pos_ids_79,
                attn_mask=attn_mask,
                past_kv=past_i_90,
                scope=scope_89,
            )
            x_83 = y_91
            new_i_93 = new_kv_92
            if new_kv_84 is None:
                new_kv_84 = None
            else:
                new_kv_84 = list(new_kv_84)
                new_kv_84.append(new_i_93)
        node_path_94 = self._join_scope(scope, "n_call_12")
        xnorm_96 = x_83.float() * torch.rsqrt(
            torch.mean(x_83.float() * x_83.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        pipe_11_95 = xnorm_96 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_94), "model.norm.weight")
            ).float()
        )
        pipe_11_95 = pipe_11_95.type_as(x_83)
        node_path_97 = self._join_scope(scope, "n_op_13")
        if pipe_11_95.numel() == 0:
            logits_98 = pipe_11_95.new_empty(
                (
                    *pipe_11_95.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_97), "model.embed_tokens.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            logits_98 = F.linear(
                pipe_11_95,
                self._param(
                    self._join_scope(self._scope_of(node_path_97), "model.embed_tokens.weight")
                ),
                None,
            )
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_98
        outputs["new_kv"] = new_kv_84
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
