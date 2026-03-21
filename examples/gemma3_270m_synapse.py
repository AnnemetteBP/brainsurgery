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
        xn_4 = xnorm_5 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_3), "input_layernorm.weight")
            ).float()
        )
        xn_4 = xn_4.type_as(x)
        node_path_6 = self._join_scope(scope, "n_op_3")
        if xn_4.numel() == 0:
            pipe_2_7 = xn_4.new_empty(
                (
                    *xn_4.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_6), "self_attn.q_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_2_7 = F.linear(
                xn_4,
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
        if xn_4.numel() == 0:
            pipe_7_16 = xn_4.new_empty(
                (
                    *xn_4.shape[:-1],
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
                xn_4,
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
        if xn_4.numel() == 0:
            pipe_12_25 = xn_4.new_empty(
                (
                    *xn_4.shape[:-1],
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
                xn_4,
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
        theta_30 = 10000.0 + (1000000.0 - 10000.0) * (((i + 1) % 6) == 0)
        if q_13.ndim != 4 or k_22.ndim != 4:
            raise ValueError("rope_pair expects q and k to be rank-4 [batch, heads, seq, head_dim]")
        if int(q_13.shape[0]) != int(k_22.shape[0]):
            raise ValueError("rope_pair expects q and k to have matching batch size")
        if int(q_13.shape[-2]) != int(k_22.shape[-2]):
            raise ValueError("rope_pair expects q and k to have matching sequence length")
        if int(q_13.shape[-1]) != int(k_22.shape[-1]):
            raise ValueError("rope_pair expects q and k to have matching head dimension")
        half_31 = q_13.shape[-1] // 2
        if int(q_13.shape[-1]) % 2 != 0:
            raise ValueError("rope_pair expects even head dimension")
        inv_freq_32 = 1.0 / (
            float(theta_30)
            ** (torch.arange(0, half_31, device=q_13.device, dtype=q_13.dtype) / float(half_31))
        )
        if pos_ids is None:
            raise ValueError("rope_pair.position_ids must not be null")
        if pos_ids.ndim != 2:
            raise ValueError("rope_pair.position_ids must be rank-2 [batch, seq]")
        if int(pos_ids.shape[0]) != int(q_13.shape[0]):
            raise ValueError("rope_pair.position_ids batch size must match q/k batch")
        if int(pos_ids.shape[1]) != int(q_13.shape[-2]):
            raise ValueError("rope_pair.position_ids width must match q/k sequence length")
        pos_33 = pos_ids.to(device=q_13.device, dtype=q_13.dtype)
        ang_34 = pos_33.unsqueeze(-1) * inv_freq_32.unsqueeze(0).unsqueeze(0)
        cos_35 = torch.cos(ang_34).unsqueeze(1)
        sin_36 = torch.sin(ang_34).unsqueeze(1)
        q1_37 = q_13[..., :half_31]
        q2_38 = q_13[..., half_31 : 2 * half_31]
        k1_39 = k_22[..., :half_31]
        k2_40 = k_22[..., half_31 : 2 * half_31]
        q_13 = torch.cat([q1_37 * cos_35 - q2_38 * sin_36, q1_37 * sin_36 + q2_38 * cos_35], dim=-1)
        k_22 = torch.cat([k1_39 * cos_35 - k2_40 * sin_36, k1_39 * sin_36 + k2_40 * cos_35], dim=-1)
        if past_kv is None:
            k_22 = k_22
            v_29 = v_29
        else:
            k_22 = torch.cat([past_kv[0], k_22], dim=-2)
            v_29 = torch.cat([past_kv[1], v_29], dim=-2)
        new_kv_41 = (k_22, v_29)
        n_rep_42 = int(4.0)
        if n_rep_42 == 1:
            k_22 = k_22
        else:
            k_22 = (
                k_22[:, :, None, :, :]
                .expand(k_22.shape[0], k_22.shape[1], n_rep_42, k_22.shape[2], k_22.shape[3])
                .reshape(k_22.shape[0], k_22.shape[1] * n_rep_42, k_22.shape[2], k_22.shape[3])
            )
        n_rep_43 = int(4.0)
        if n_rep_43 == 1:
            v_29 = v_29
        else:
            v_29 = (
                v_29[:, :, None, :, :]
                .expand(v_29.shape[0], v_29.shape[1], n_rep_43, v_29.shape[2], v_29.shape[3])
                .reshape(v_29.shape[0], v_29.shape[1] * n_rep_43, v_29.shape[2], v_29.shape[3])
            )
        window_44 = 512 + (32768 - 512) * (((i + 1) % 6) == 0)
        q_len_46 = q_13.shape[-2]
        k_len_47 = k_22.shape[-2]
        if not hasattr(self, "_causal_mask_cache"):
            self._causal_mask_cache = {}
        window_54 = int(window_44)
        if attn_mask is None:
            pad_key_55 = None
        else:
            pad_key_55 = (
                int(attn_mask.data_ptr()),
                int(attn_mask.storage_offset()),
                tuple(int(x) for x in attn_mask.shape),
            )
        cache_key_48 = (
            int(q_len_46),
            int(k_len_47),
            window_54,
            q_13.dtype,
            q_13.device,
            pad_key_55,
        )
        cached_mask_49 = self._causal_mask_cache.get(cache_key_48)
        if torch.is_tensor(cached_mask_49):
            mask_45 = cached_mask_49
        else:
            j_idx_51 = torch.arange(k_len_47, device=q_13.device).unsqueeze(0)
            if q_len_46 == 1:
                keep_52 = j_idx_51 >= (k_len_47 - window_54)
            else:
                i_idx_50 = torch.arange(q_len_46, device=q_13.device).unsqueeze(1)
                keep_52 = j_idx_51 <= i_idx_50
                keep_52 = keep_52 & (j_idx_51 >= (i_idx_50 - window_54 + 1))
            if attn_mask is not None:
                if attn_mask.ndim != 2:
                    raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
                if int(attn_mask.shape[-1]) != k_len_47:
                    raise ValueError(
                        "causal_mask.padding_mask width must match key sequence length"
                    )
                pad_keep_56 = attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
                keep_52 = keep_52.unsqueeze(0).unsqueeze(0) & pad_keep_56
            else:
                keep_52 = keep_52.view(1, 1, q_len_46, k_len_47)
            mask_val_53 = torch.finfo(q_13.dtype).min
            mask_45 = torch.where(
                keep_52,
                torch.zeros((), dtype=q_13.dtype, device=q_13.device),
                torch.full((), mask_val_53, dtype=q_13.dtype, device=q_13.device),
            )
            self._causal_mask_cache[cache_key_48] = mask_45
        pipe_22_57 = F.scaled_dot_product_attention(
            q_13,
            k_22,
            v_29,
            attn_mask=mask_45,
            dropout_p=0.0,
            is_causal=(q_13.shape[-2] > 1 and mask_45 is None),
            scale=0.0625,
        )
        pipe_24_58 = (
            pipe_22_57.transpose(1, 2)
            .contiguous()
            .view(
                pipe_22_57.shape[0], pipe_22_57.shape[2], pipe_22_57.shape[1] * pipe_22_57.shape[3]
            )
        )
        node_path_59 = self._join_scope(scope, "n_op_26")
        if pipe_24_58.numel() == 0:
            a_60 = pipe_24_58.new_empty(
                (
                    *pipe_24_58.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_59), "self_attn.o_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            a_60 = F.linear(
                pipe_24_58,
                self._param(
                    self._join_scope(self._scope_of(node_path_59), "self_attn.o_proj.weight")
                ),
                None,
            )
        node_path_61 = self._join_scope(scope, "n_call_27")
        xnorm_62 = a_60.float() * torch.rsqrt(
            torch.mean(a_60.float() * a_60.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        a_60 = xnorm_62 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_61), "post_attention_layernorm.weight")
            ).float()
        )
        a_60 = a_60.type_as(a_60)
        x = x + a_60
        node_path_63 = self._join_scope(scope, "n_call_29")
        xnorm_64 = x.float() * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        xn_4 = xnorm_64 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_63), "pre_feedforward_layernorm.weight")
            ).float()
        )
        xn_4 = xn_4.type_as(x)
        node_path_65 = self._join_scope(scope, "n_op_30")
        if xn_4.numel() == 0:
            g_66 = xn_4.new_empty(
                (
                    *xn_4.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_65), "mlp.gate_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            g_66 = F.linear(
                xn_4,
                self._param(self._join_scope(self._scope_of(node_path_65), "mlp.gate_proj.weight")),
                None,
            )
        node_path_67 = self._join_scope(scope, "n_op_31")
        if xn_4.numel() == 0:
            u_68 = xn_4.new_empty(
                (
                    *xn_4.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_67), "mlp.up_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            u_68 = F.linear(
                xn_4,
                self._param(self._join_scope(self._scope_of(node_path_67), "mlp.up_proj.weight")),
                None,
            )
        pipe_32_69 = (
            0.5
            * g_66
            * (1.0 + torch.tanh(0.7978845608028654 * (g_66 + 0.044715 * g_66 * g_66 * g_66)))
        )
        pipe_34_70 = pipe_32_69 * u_68
        node_path_71 = self._join_scope(scope, "n_op_36")
        if pipe_34_70.numel() == 0:
            out_0_72 = pipe_34_70.new_empty(
                (
                    *pipe_34_70.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_71), "mlp.down_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            out_0_72 = F.linear(
                pipe_34_70,
                self._param(self._join_scope(self._scope_of(node_path_71), "mlp.down_proj.weight")),
                None,
            )
        m_73 = out_0_72
        node_path_74 = self._join_scope(scope, "n_call_38")
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
        out_0_72 = x + m_73
        return (out_0_72, new_kv_41)

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
        out_0_76 = self._block_Cache_past_length(cache=past_kv, scope=scope)
        kwarg_1_77 = out_0_76
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
            full_pos_80 = attn_mask.to(torch.long).cumsum(dim=-1) - 1
            full_pos_80 = full_pos_80.masked_fill(attn_mask == 0, 0)
            pos_ids_78 = full_pos_80[:, -input_ids.shape[1] :]
        else:
            pos_offset_79 = int(kwarg_1_77)
            if pos_offset_79 < 0:
                raise ValueError("position_ids.past_length must resolve to non-negative int")
            pos_ids_78 = torch.arange(
                pos_offset_79,
                pos_offset_79 + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
        node_path_81 = self._join_scope(scope, "n_op_4")
        x_82 = F.embedding(
            input_ids,
            emitter._param(
                self._join_scope(self._scope_of(node_path_81), "model.embed_tokens.weight")
            ),
        )
        x_82 = x_82 * torch.tensor(float(25.298221281347036), dtype=x_82.dtype, device=x_82.device)
        new_kv_83 = None
        if use_cache:
            new_kv_83 = []
        if not (use_cache):
            new_kv_83 = None
        to_85 = int(18)
        from_86 = int(0)
        step_87 = int(1)
        for i_84 in self._for_values(from_value=from_86, to_value=to_85, step_value=step_87):
            scope_88 = self._join_scope(scope, f"model.layers.{i_84}")
            if past_kv is None:
                past_i_89 = None
            else:
                try:
                    past_i_89 = past_kv[int(i_84)]
                except (IndexError, KeyError, TypeError):
                    past_i_89 = None
            out_0_90, new_kv_91 = self._block_gemma3_block(
                x=x_82,
                i=i_84,
                pos_ids=pos_ids_78,
                attn_mask=attn_mask,
                past_kv=past_i_89,
                scope=scope_88,
            )
            x_82 = out_0_90
            new_i_92 = new_kv_91
            if new_kv_83 is None:
                new_kv_83 = None
            else:
                new_kv_83 = list(new_kv_83)
                new_kv_83.append(new_i_92)
        node_path_93 = self._join_scope(scope, "n_call_12")
        xnorm_95 = x_82.float() * torch.rsqrt(
            torch.mean(x_82.float() * x_82.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        pipe_11_94 = xnorm_95 * (
            1.0
            + emitter._param(
                self._join_scope(self._scope_of(node_path_93), "model.norm.weight")
            ).float()
        )
        pipe_11_94 = pipe_11_94.type_as(x_82)
        node_path_96 = self._join_scope(scope, "n_op_13")
        if pipe_11_94.numel() == 0:
            logits_97 = pipe_11_94.new_empty(
                (
                    *pipe_11_94.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_96), "model.embed_tokens.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            logits_97 = F.linear(
                pipe_11_94,
                self._param(
                    self._join_scope(self._scope_of(node_path_96), "model.embed_tokens.weight")
                ),
                None,
            )
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_97
        outputs["new_kv"] = new_kv_83
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
