from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class OLMoE1B7B0924Synapse(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int | float | bool] = {
            "D": 2048,
            "V": 50304,
            "L": 16,
            "H": 16,
            "HD": 128,
            "E": 64,
            "EPT": 8,
            "C": 4096,
            "EPS": 1e-05,
            "THETA": 10000.0,
            "ATTN_SCALE": 0.08838834764831845,
        }
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "OLMoE1B7B0924Synapse":
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

    def _block_Cache_update(self, past, k, v, use_cache, scope: str) -> tuple[Any, ...]:
        env: dict[str, Any] = {}
        env["past"] = past
        env["k"] = k
        env["v"] = v
        env["use_cache"] = use_cache
        if past is None:
            k_all_1 = k
            v_all_2 = v
        else:
            k_all_1 = torch.cat([past[0], k], dim=-2)
            v_all_2 = torch.cat([past[1], v], dim=-2)
        present_3 = (k_all_1, v_all_2)
        k_ctx_4 = None
        if use_cache:
            k_ctx_4 = k_all_1
        if not (use_cache):
            k_ctx_4 = k
        v_ctx_5 = None
        if use_cache:
            v_ctx_5 = v_all_2
        if not (use_cache):
            v_ctx_5 = v
        return (k_ctx_4, v_ctx_5, present_3)

    def _block_Cache_past_length(self, cache, scope: str) -> tuple[Any, ...]:
        env: dict[str, Any] = {}
        env["cache"] = cache
        first_6 = None if cache is None else cache[int(0)]
        if first_6 is None:
            out_0_7 = 0
        else:
            out_0_7 = int(first_6[0].shape[-2])
        return out_0_7

    def _block_expert_ffn(self, x, scope: str) -> tuple[Any, ...]:
        env: dict[str, Any] = {}
        env["x"] = x
        node_path_8 = self._join_scope(scope, "n_op_1")
        if x.numel() == 0:
            gate_9 = x.new_empty(
                (
                    *x.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_8), "gate_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            gate_9 = F.linear(
                x,
                self._param(self._join_scope(self._scope_of(node_path_8), "gate_proj.weight")),
                None,
            )
        node_path_10 = self._join_scope(scope, "n_op_2")
        if x.numel() == 0:
            up_11 = x.new_empty(
                (
                    *x.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_10), "up_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            up_11 = F.linear(
                x,
                self._param(self._join_scope(self._scope_of(node_path_10), "up_proj.weight")),
                None,
            )
        pipe_3_12 = F.silu(gate_9)
        pipe_5_13 = pipe_3_12 * up_11
        node_path_14 = self._join_scope(scope, "n_op_7")
        if pipe_5_13.numel() == 0:
            out_0_15 = pipe_5_13.new_empty(
                (
                    *pipe_5_13.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_14), "down_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            out_0_15 = F.linear(
                pipe_5_13,
                self._param(self._join_scope(self._scope_of(node_path_14), "down_proj.weight")),
                None,
            )
        return out_0_15

    def _block_olmoe_decoder_block(
        self, x, pos_ids, attn_mask, past_kv, use_cache, scope: str
    ) -> tuple[Any, ...]:
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        env["pos_ids"] = pos_ids
        env["attn_mask"] = attn_mask
        env["past_kv"] = past_kv
        env["use_cache"] = use_cache
        node_path_16 = self._join_scope(scope, "n_call_1")
        xnorm_18 = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + float(1e-05))
        x_norm_17 = xnorm_18 * emitter._param(
            self._join_scope(self._scope_of(node_path_16), "input_layernorm.weight")
        )
        node_path_19 = self._join_scope(scope, "n_op_3")
        if x_norm_17.numel() == 0:
            pipe_2_20 = x_norm_17.new_empty(
                (
                    *x_norm_17.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_19), "self_attn.q_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_2_20 = F.linear(
                x_norm_17,
                self._param(
                    self._join_scope(self._scope_of(node_path_19), "self_attn.q_proj.weight")
                ),
                None,
            )
        node_path_21 = self._join_scope(scope, "n_call_5")
        xnorm_23 = pipe_2_20 * torch.rsqrt(
            torch.mean(pipe_2_20 * pipe_2_20, dim=-1, keepdim=True) + float(1e-05)
        )
        pipe_4_22 = xnorm_23 * emitter._param(
            self._join_scope(self._scope_of(node_path_21), "self_attn.q_norm.weight")
        )
        heads_24 = int(16)
        head_dim_25 = None
        if heads_24 is None and head_dim_25 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_24 is None:
            if pipe_4_22.shape[-1] % int(head_dim_25) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_24 = pipe_4_22.shape[-1] // int(head_dim_25)
        if head_dim_25 is None:
            if pipe_4_22.shape[-1] % int(heads_24) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_25 = pipe_4_22.shape[-1] // int(heads_24)
        expected_hidden_26 = int(heads_24) * int(head_dim_25)
        if pipe_4_22.shape[-1] != expected_hidden_26:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        q_27 = pipe_4_22.view(
            pipe_4_22.shape[0], pipe_4_22.shape[1], int(heads_24), int(head_dim_25)
        ).transpose(1, 2)
        node_path_28 = self._join_scope(scope, "n_op_8")
        if x_norm_17.numel() == 0:
            pipe_7_29 = x_norm_17.new_empty(
                (
                    *x_norm_17.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_28), "self_attn.k_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_7_29 = F.linear(
                x_norm_17,
                self._param(
                    self._join_scope(self._scope_of(node_path_28), "self_attn.k_proj.weight")
                ),
                None,
            )
        node_path_30 = self._join_scope(scope, "n_call_10")
        xnorm_32 = pipe_7_29 * torch.rsqrt(
            torch.mean(pipe_7_29 * pipe_7_29, dim=-1, keepdim=True) + float(1e-05)
        )
        pipe_9_31 = xnorm_32 * emitter._param(
            self._join_scope(self._scope_of(node_path_30), "self_attn.k_norm.weight")
        )
        heads_33 = int(16)
        head_dim_34 = None
        if heads_33 is None and head_dim_34 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_33 is None:
            if pipe_9_31.shape[-1] % int(head_dim_34) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_33 = pipe_9_31.shape[-1] // int(head_dim_34)
        if head_dim_34 is None:
            if pipe_9_31.shape[-1] % int(heads_33) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_34 = pipe_9_31.shape[-1] // int(heads_33)
        expected_hidden_35 = int(heads_33) * int(head_dim_34)
        if pipe_9_31.shape[-1] != expected_hidden_35:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        k_36 = pipe_9_31.view(
            pipe_9_31.shape[0], pipe_9_31.shape[1], int(heads_33), int(head_dim_34)
        ).transpose(1, 2)
        node_path_37 = self._join_scope(scope, "n_op_13")
        if x_norm_17.numel() == 0:
            pipe_12_38 = x_norm_17.new_empty(
                (
                    *x_norm_17.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_37), "self_attn.v_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_12_38 = F.linear(
                x_norm_17,
                self._param(
                    self._join_scope(self._scope_of(node_path_37), "self_attn.v_proj.weight")
                ),
                None,
            )
        heads_39 = int(16)
        head_dim_40 = None
        if heads_39 is None and head_dim_40 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_39 is None:
            if pipe_12_38.shape[-1] % int(head_dim_40) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_39 = pipe_12_38.shape[-1] // int(head_dim_40)
        if head_dim_40 is None:
            if pipe_12_38.shape[-1] % int(heads_39) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_40 = pipe_12_38.shape[-1] // int(heads_39)
        expected_hidden_41 = int(heads_39) * int(head_dim_40)
        if pipe_12_38.shape[-1] != expected_hidden_41:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        v_42 = pipe_12_38.view(
            pipe_12_38.shape[0], pipe_12_38.shape[1], int(heads_39), int(head_dim_40)
        ).transpose(1, 2)
        half_45 = q_27.shape[-1] // 2
        inv_freq_46 = 1.0 / (
            float(10000.0)
            ** (torch.arange(0, half_45, device=q_27.device, dtype=q_27.dtype) / float(half_45))
        )
        if pos_ids is not None:
            if pos_ids.ndim != 2:
                raise ValueError("apply_rope_pair.position_ids must be rank-2 [batch, seq]")
            if int(pos_ids.shape[0]) != int(q_27.shape[0]):
                raise ValueError("apply_rope_pair.position_ids batch size must match q/k batch")
            if int(pos_ids.shape[1]) != int(q_27.shape[-2]):
                raise ValueError(
                    "apply_rope_pair.position_ids width must match q/k sequence length"
                )
            pos_47 = pos_ids.to(device=q_27.device, dtype=q_27.dtype)
            ang_48 = pos_47.unsqueeze(-1) * inv_freq_46.unsqueeze(0).unsqueeze(0)
            cos_49 = torch.cos(ang_48).unsqueeze(1)
            sin_50 = torch.sin(ang_48).unsqueeze(1)
        else:
            pos_47 = torch.arange(
                int(0), int(0) + q_27.shape[-2], device=q_27.device, dtype=q_27.dtype
            )
            ang_48 = torch.einsum("t,d->td", pos_47, inv_freq_46)
            cos_49 = torch.cos(ang_48)[None, None, :, :]
            sin_50 = torch.sin(ang_48)[None, None, :, :]
        q1_51 = q_27[..., :half_45]
        q2_52 = q_27[..., half_45 : 2 * half_45]
        k1_53 = k_36[..., :half_45]
        k2_54 = k_36[..., half_45 : 2 * half_45]
        qr_43 = torch.cat(
            [q1_51 * cos_49 - q2_52 * sin_50, q1_51 * sin_50 + q2_52 * cos_49], dim=-1
        )
        kr_44 = torch.cat(
            [k1_53 * cos_49 - k2_54 * sin_50, k1_53 * sin_50 + k2_54 * cos_49], dim=-1
        )
        k_ctx_55, v_ctx_56, present_57 = self._block_Cache_update(
            past=past_kv, k=kr_44, v=v_42, use_cache=use_cache, scope=scope
        )
        k_ctx_58 = k_ctx_55
        v_ctx_59 = v_ctx_56
        present_kv_60 = present_57
        q_len_62 = qr_43.shape[-2]
        k_len_63 = k_ctx_58.shape[-2]
        if not hasattr(self, "_causal_mask_cache"):
            self._causal_mask_cache = {}
        window_70 = int(4096)
        if attn_mask is None:
            pad_key_71 = None
        else:
            pad_key_71 = (
                int(attn_mask.data_ptr()),
                int(attn_mask.storage_offset()),
                tuple(int(x) for x in attn_mask.shape),
            )
        cache_key_64 = (
            int(q_len_62),
            int(k_len_63),
            window_70,
            qr_43.dtype,
            qr_43.device,
            pad_key_71,
        )
        cached_mask_65 = self._causal_mask_cache.get(cache_key_64)
        if torch.is_tensor(cached_mask_65):
            mask_61 = cached_mask_65
        else:
            j_idx_67 = torch.arange(k_len_63, device=qr_43.device).unsqueeze(0)
            if q_len_62 == 1:
                keep_68 = j_idx_67 >= (k_len_63 - window_70)
            else:
                i_idx_66 = torch.arange(q_len_62, device=qr_43.device).unsqueeze(1)
                keep_68 = j_idx_67 <= i_idx_66
                keep_68 = keep_68 & (j_idx_67 >= (i_idx_66 - window_70 + 1))
            if attn_mask is not None:
                if attn_mask.ndim != 2:
                    raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
                if int(attn_mask.shape[-1]) != k_len_63:
                    raise ValueError(
                        "causal_mask.padding_mask width must match key sequence length"
                    )
                pad_keep_72 = attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
                keep_68 = keep_68.unsqueeze(0).unsqueeze(0) & pad_keep_72
            else:
                keep_68 = keep_68.view(1, 1, q_len_62, k_len_63)
            mask_val_69 = torch.finfo(qr_43.dtype).min
            mask_61 = torch.where(
                keep_68,
                torch.zeros((), dtype=qr_43.dtype, device=qr_43.device),
                torch.full((), mask_val_69, dtype=qr_43.dtype, device=qr_43.device),
            )
            self._causal_mask_cache[cache_key_64] = mask_61
        pipe_18_73 = F.scaled_dot_product_attention(
            qr_43,
            k_ctx_58,
            v_ctx_59,
            attn_mask=mask_61,
            dropout_p=0.0,
            is_causal=(qr_43.shape[-2] > 1 and mask_61 is None),
            scale=0.08838834764831845,
        )
        a_74 = (
            pipe_18_73.transpose(1, 2)
            .contiguous()
            .view(
                pipe_18_73.shape[0], pipe_18_73.shape[2], pipe_18_73.shape[1] * pipe_18_73.shape[3]
            )
        )
        node_path_75 = self._join_scope(scope, "n_op_21")
        if a_74.numel() == 0:
            a_74 = a_74.new_empty(
                (
                    *a_74.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_75), "self_attn.o_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            a_74 = F.linear(
                a_74,
                self._param(
                    self._join_scope(self._scope_of(node_path_75), "self_attn.o_proj.weight")
                ),
                None,
            )
        x = x + a_74
        node_path_76 = self._join_scope(scope, "n_call_23")
        xnorm_78 = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + float(1e-05))
        xn_77 = xnorm_78 * emitter._param(
            self._join_scope(self._scope_of(node_path_76), "post_attention_layernorm.weight")
        )
        node_path_79 = self._join_scope(scope, "n_op_25")
        if xn_77.numel() == 0:
            pipe_24_80 = xn_77.new_empty(
                (
                    *xn_77.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_79), "mlp.gate.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_24_80 = F.linear(
                xn_77,
                self._param(self._join_scope(self._scope_of(node_path_79), "mlp.gate.weight")),
                None,
            )
        pipe_26_81 = F.softmax(pipe_24_80, dim=int(-1), dtype=torch.float32)
        topk_scores_82, topk_indices_83 = torch.topk(
            pipe_26_81, int(8), dim=int(-1), largest=True, sorted=True
        )
        m_84 = torch.zeros_like(xn_77)
        to_86 = int(64)
        from_87 = int(0)
        step_88 = int(1)
        for e_85 in self._for_values(from_value=from_87, to_value=to_86, step_value=step_88):
            scope_89 = self._join_scope(scope, f"mlp.experts.{e_85}")
            hidden_flat_94 = xn_77.reshape(-1, xn_77.shape[-1])
            topk_scores_flat_95 = topk_scores_82.reshape(-1, topk_scores_82.shape[-1])
            topk_indices_flat_96 = topk_indices_83.reshape(-1, topk_indices_83.shape[-1])
            expert_pos_97 = (topk_indices_flat_96 == int(e_85)).nonzero(as_tuple=False)
            token_idx_91 = expert_pos_97[:, 0]
            topk_pos_92 = expert_pos_97[:, 1]
            x_sel_90 = hidden_flat_94[token_idx_91]
            sel_scores_93 = topk_scores_flat_95[token_idx_91, topk_pos_92].to(x_sel_90.dtype)
            out_0_98 = self._block_expert_ffn(x=x_sel_90, scope=scope_89)
            x_upd_99 = out_0_98
            m_84 = m_84
            if token_idx_91.numel() != 0:
                _acc = m_84.reshape(-1, m_84.shape[-1])
                _upd = x_upd_99 * sel_scores_93.unsqueeze(-1).to(x_upd_99.dtype)
                _acc.index_add_(0, token_idx_91, _upd.to(_acc.dtype))
        out_0_100 = x + m_84
        return (out_0_100, present_kv_60)

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        input_specs = {
            "input_ids": {"optional": False},
            "attn_mask": {"optional": True},
            "past_key_values": {"optional": True},
            "use_cache": {"optional": True},
        }
        env = self._prepare_env(input_ids, inputs, input_specs)
        scope = ""
        emitter = self
        input_ids = self._safe_get(env, "input_ids")
        attn_mask = env.get("attn_mask")
        past_key_values = env.get("past_key_values")
        use_cache = env.get("use_cache")
        out_0_101 = self._block_Cache_past_length(cache=past_key_values, scope=scope)
        kwarg_1_102 = out_0_101
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
            full_pos_105 = attn_mask.to(torch.long).cumsum(dim=-1) - 1
            full_pos_105 = full_pos_105.masked_fill(attn_mask == 0, 0)
            pos_ids_103 = full_pos_105[:, -input_ids.shape[1] :]
        else:
            pos_offset_104 = int(kwarg_1_102)
            if pos_offset_104 < 0:
                raise ValueError("position_ids.past_length must resolve to non-negative int")
            pos_ids_103 = torch.arange(
                pos_offset_104,
                pos_offset_104 + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
        node_path_106 = self._join_scope(scope, "n_op_4")
        x_107 = F.embedding(
            input_ids,
            emitter._param(
                self._join_scope(self._scope_of(node_path_106), "model.embed_tokens.weight")
            ),
        )
        present_key_values_108 = []
        to_110 = int(16)
        from_111 = int(0)
        step_112 = int(1)
        for i_109 in self._for_values(from_value=from_111, to_value=to_110, step_value=step_112):
            scope_113 = self._join_scope(scope, f"model.layers.{i_109}")
            past_i_114 = None if past_key_values is None else past_key_values[int(i_109)]
            out_0_115, present_kv_116 = self._block_olmoe_decoder_block(
                x=x_107,
                pos_ids=pos_ids_103,
                attn_mask=attn_mask,
                past_kv=past_i_114,
                use_cache=use_cache,
                scope=scope_113,
            )
            x_107 = out_0_115
            present_i_117 = present_kv_116
            if use_cache:
                present_key_values_108 = list(present_key_values_108)
                present_key_values_108.append(present_i_117)
            if not (use_cache):
                present_key_values_108 = present_key_values_108
        node_path_118 = self._join_scope(scope, "n_call_11")
        xnorm_119 = x_107 * torch.rsqrt(
            torch.mean(x_107 * x_107, dim=-1, keepdim=True) + float(1e-05)
        )
        x_107 = xnorm_119 * emitter._param(
            self._join_scope(self._scope_of(node_path_118), "model.norm.weight")
        )
        node_path_120 = self._join_scope(scope, "n_op_12")
        if x_107.numel() == 0:
            logits_121 = x_107.new_empty(
                (
                    *x_107.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_120), "lm_head.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            logits_121 = F.linear(
                x_107,
                self._param(self._join_scope(self._scope_of(node_path_120), "lm_head.weight")),
                None,
            )
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_121
        outputs["present_key_values"] = present_key_values_108
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
        past_key_values = None
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        cur_len = start_len
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                while cur_len < max_len and not torch.all(finished):
                    step_input = (
                        generated[:, :cur_len]
                        if past_key_values is None
                        else generated[:, cur_len - 1 : cur_len]
                    )
                    if generated_mask is None:
                        model_out = self.forward(
                            step_input, past_key_values=past_key_values, use_cache=True
                        )
                    else:
                        model_out = self.forward(
                            step_input,
                            attention_mask=generated_mask[:, :cur_len],
                            attn_mask=generated_mask[:, :cur_len],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    if isinstance(model_out, dict):
                        logits = model_out["logits"]
                        if "past_key_values" in model_out:
                            past_key_values = model_out["past_key_values"]
                        elif "present_key_values" in model_out:
                            past_key_values = model_out["present_key_values"]
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
