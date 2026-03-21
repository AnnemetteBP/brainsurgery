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

    def _block_expert_ffn(self, x, scope: str) -> tuple[Any, ...]:
        env: dict[str, Any] = {}
        env["x"] = x
        node_path_3 = self._join_scope(scope, "n_op_1")
        if x.numel() == 0:
            gate_4 = x.new_empty(
                (
                    *x.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_3), "gate_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            gate_4 = F.linear(
                x,
                self._param(self._join_scope(self._scope_of(node_path_3), "gate_proj.weight")),
                None,
            )
        node_path_5 = self._join_scope(scope, "n_op_2")
        if x.numel() == 0:
            up_6 = x.new_empty(
                (
                    *x.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_5), "up_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            up_6 = F.linear(
                x,
                self._param(self._join_scope(self._scope_of(node_path_5), "up_proj.weight")),
                None,
            )
        pipe_3_7 = F.silu(gate_4)
        pipe_5_8 = pipe_3_7 * up_6
        node_path_9 = self._join_scope(scope, "n_op_7")
        if pipe_5_8.numel() == 0:
            out_0_10 = pipe_5_8.new_empty(
                (
                    *pipe_5_8.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_9), "down_proj.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            out_0_10 = F.linear(
                pipe_5_8,
                self._param(self._join_scope(self._scope_of(node_path_9), "down_proj.weight")),
                None,
            )
        return out_0_10

    def _block_olmoe_decoder_block(
        self, x, pos_ids, attn_mask, past_kv, scope: str
    ) -> tuple[Any, ...]:
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        env["pos_ids"] = pos_ids
        env["attn_mask"] = attn_mask
        env["past_kv"] = past_kv
        node_path_11 = self._join_scope(scope, "n_call_1")
        xnorm_13 = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + float(1e-05))
        x_norm_12 = xnorm_13 * emitter._param(
            self._join_scope(self._scope_of(node_path_11), "input_layernorm.weight")
        )
        node_path_14 = self._join_scope(scope, "n_op_3")
        if x_norm_12.numel() == 0:
            pipe_2_15 = x_norm_12.new_empty(
                (
                    *x_norm_12.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_14), "self_attn.q_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_2_15 = F.linear(
                x_norm_12,
                self._param(
                    self._join_scope(self._scope_of(node_path_14), "self_attn.q_proj.weight")
                ),
                None,
            )
        node_path_16 = self._join_scope(scope, "n_call_5")
        xnorm_18 = pipe_2_15 * torch.rsqrt(
            torch.mean(pipe_2_15 * pipe_2_15, dim=-1, keepdim=True) + float(1e-05)
        )
        pipe_4_17 = xnorm_18 * emitter._param(
            self._join_scope(self._scope_of(node_path_16), "self_attn.q_norm.weight")
        )
        heads_19 = int(16)
        head_dim_20 = None
        if heads_19 is None and head_dim_20 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_19 is None:
            if pipe_4_17.shape[-1] % int(head_dim_20) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_19 = pipe_4_17.shape[-1] // int(head_dim_20)
        if head_dim_20 is None:
            if pipe_4_17.shape[-1] % int(heads_19) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_20 = pipe_4_17.shape[-1] // int(heads_19)
        expected_hidden_21 = int(heads_19) * int(head_dim_20)
        if pipe_4_17.shape[-1] != expected_hidden_21:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        q_22 = pipe_4_17.view(
            pipe_4_17.shape[0], pipe_4_17.shape[1], int(heads_19), int(head_dim_20)
        ).transpose(1, 2)
        node_path_23 = self._join_scope(scope, "n_op_8")
        if x_norm_12.numel() == 0:
            pipe_7_24 = x_norm_12.new_empty(
                (
                    *x_norm_12.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_23), "self_attn.k_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_7_24 = F.linear(
                x_norm_12,
                self._param(
                    self._join_scope(self._scope_of(node_path_23), "self_attn.k_proj.weight")
                ),
                None,
            )
        node_path_25 = self._join_scope(scope, "n_call_10")
        xnorm_27 = pipe_7_24 * torch.rsqrt(
            torch.mean(pipe_7_24 * pipe_7_24, dim=-1, keepdim=True) + float(1e-05)
        )
        pipe_9_26 = xnorm_27 * emitter._param(
            self._join_scope(self._scope_of(node_path_25), "self_attn.k_norm.weight")
        )
        heads_28 = int(16)
        head_dim_29 = None
        if heads_28 is None and head_dim_29 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_28 is None:
            if pipe_9_26.shape[-1] % int(head_dim_29) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_28 = pipe_9_26.shape[-1] // int(head_dim_29)
        if head_dim_29 is None:
            if pipe_9_26.shape[-1] % int(heads_28) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_29 = pipe_9_26.shape[-1] // int(heads_28)
        expected_hidden_30 = int(heads_28) * int(head_dim_29)
        if pipe_9_26.shape[-1] != expected_hidden_30:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        k_31 = pipe_9_26.view(
            pipe_9_26.shape[0], pipe_9_26.shape[1], int(heads_28), int(head_dim_29)
        ).transpose(1, 2)
        node_path_32 = self._join_scope(scope, "n_op_13")
        if x_norm_12.numel() == 0:
            pipe_12_33 = x_norm_12.new_empty(
                (
                    *x_norm_12.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_32), "self_attn.v_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_12_33 = F.linear(
                x_norm_12,
                self._param(
                    self._join_scope(self._scope_of(node_path_32), "self_attn.v_proj.weight")
                ),
                None,
            )
        heads_34 = int(16)
        head_dim_35 = None
        if heads_34 is None and head_dim_35 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_34 is None:
            if pipe_12_33.shape[-1] % int(head_dim_35) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_34 = pipe_12_33.shape[-1] // int(head_dim_35)
        if head_dim_35 is None:
            if pipe_12_33.shape[-1] % int(heads_34) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_35 = pipe_12_33.shape[-1] // int(heads_34)
        expected_hidden_36 = int(heads_34) * int(head_dim_35)
        if pipe_12_33.shape[-1] != expected_hidden_36:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        v_37 = pipe_12_33.view(
            pipe_12_33.shape[0], pipe_12_33.shape[1], int(heads_34), int(head_dim_35)
        ).transpose(1, 2)
        if q_22.ndim != 4 or k_31.ndim != 4:
            raise ValueError("rope_pair expects q and k to be rank-4 [batch, heads, seq, head_dim]")
        if int(q_22.shape[0]) != int(k_31.shape[0]):
            raise ValueError("rope_pair expects q and k to have matching batch size")
        if int(q_22.shape[-2]) != int(k_31.shape[-2]):
            raise ValueError("rope_pair expects q and k to have matching sequence length")
        if int(q_22.shape[-1]) != int(k_31.shape[-1]):
            raise ValueError("rope_pair expects q and k to have matching head dimension")
        half_40 = q_22.shape[-1] // 2
        if int(q_22.shape[-1]) % 2 != 0:
            raise ValueError("rope_pair expects even head dimension")
        inv_freq_41 = 1.0 / (
            float(10000.0)
            ** (torch.arange(0, half_40, device=q_22.device, dtype=q_22.dtype) / float(half_40))
        )
        if pos_ids is None:
            raise ValueError("rope_pair.position_ids must not be null")
        if pos_ids.ndim != 2:
            raise ValueError("rope_pair.position_ids must be rank-2 [batch, seq]")
        if int(pos_ids.shape[0]) != int(q_22.shape[0]):
            raise ValueError("rope_pair.position_ids batch size must match q/k batch")
        if int(pos_ids.shape[1]) != int(q_22.shape[-2]):
            raise ValueError("rope_pair.position_ids width must match q/k sequence length")
        pos_42 = pos_ids.to(device=q_22.device, dtype=q_22.dtype)
        ang_43 = pos_42.unsqueeze(-1) * inv_freq_41.unsqueeze(0).unsqueeze(0)
        cos_44 = torch.cos(ang_43).unsqueeze(1)
        sin_45 = torch.sin(ang_43).unsqueeze(1)
        q1_46 = q_22[..., :half_40]
        q2_47 = q_22[..., half_40 : 2 * half_40]
        k1_48 = k_31[..., :half_40]
        k2_49 = k_31[..., half_40 : 2 * half_40]
        qr_38 = torch.cat(
            [q1_46 * cos_44 - q2_47 * sin_45, q1_46 * sin_45 + q2_47 * cos_44], dim=-1
        )
        kr_39 = torch.cat(
            [k1_48 * cos_44 - k2_49 * sin_45, k1_48 * sin_45 + k2_49 * cos_44], dim=-1
        )
        if past_kv is None:
            k_ctx_50 = kr_39
            v_ctx_51 = v_37
        else:
            k_ctx_50 = torch.cat([past_kv[0], kr_39], dim=-2)
            v_ctx_51 = torch.cat([past_kv[1], v_37], dim=-2)
        new_kv_52 = (k_ctx_50, v_ctx_51)
        q_len_54 = qr_38.shape[-2]
        k_len_55 = k_ctx_50.shape[-2]
        if not hasattr(self, "_causal_mask_cache"):
            self._causal_mask_cache = {}
        window_62 = int(4096)
        if attn_mask is None:
            pad_key_63 = None
        else:
            pad_key_63 = (
                int(attn_mask.data_ptr()),
                int(attn_mask.storage_offset()),
                tuple(int(x) for x in attn_mask.shape),
            )
        cache_key_56 = (
            int(q_len_54),
            int(k_len_55),
            window_62,
            qr_38.dtype,
            qr_38.device,
            pad_key_63,
        )
        cached_mask_57 = self._causal_mask_cache.get(cache_key_56)
        if torch.is_tensor(cached_mask_57):
            mask_53 = cached_mask_57
        else:
            j_idx_59 = torch.arange(k_len_55, device=qr_38.device).unsqueeze(0)
            if q_len_54 == 1:
                keep_60 = j_idx_59 >= (k_len_55 - window_62)
            else:
                i_idx_58 = torch.arange(q_len_54, device=qr_38.device).unsqueeze(1)
                keep_60 = j_idx_59 <= i_idx_58
                keep_60 = keep_60 & (j_idx_59 >= (i_idx_58 - window_62 + 1))
            if attn_mask is not None:
                if attn_mask.ndim != 2:
                    raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
                if int(attn_mask.shape[-1]) != k_len_55:
                    raise ValueError(
                        "causal_mask.padding_mask width must match key sequence length"
                    )
                pad_keep_64 = attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
                keep_60 = keep_60.unsqueeze(0).unsqueeze(0) & pad_keep_64
            else:
                keep_60 = keep_60.view(1, 1, q_len_54, k_len_55)
            mask_val_61 = torch.finfo(qr_38.dtype).min
            mask_53 = torch.where(
                keep_60,
                torch.zeros((), dtype=qr_38.dtype, device=qr_38.device),
                torch.full((), mask_val_61, dtype=qr_38.dtype, device=qr_38.device),
            )
            self._causal_mask_cache[cache_key_56] = mask_53
        pipe_18_65 = F.scaled_dot_product_attention(
            qr_38,
            k_ctx_50,
            v_ctx_51,
            attn_mask=mask_53,
            dropout_p=0.0,
            is_causal=(qr_38.shape[-2] > 1 and mask_53 is None),
            scale=0.08838834764831845,
        )
        a_66 = (
            pipe_18_65.transpose(1, 2)
            .contiguous()
            .view(
                pipe_18_65.shape[0], pipe_18_65.shape[2], pipe_18_65.shape[1] * pipe_18_65.shape[3]
            )
        )
        node_path_67 = self._join_scope(scope, "n_op_21")
        if a_66.numel() == 0:
            a_66 = a_66.new_empty(
                (
                    *a_66.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(
                                self._scope_of(node_path_67), "self_attn.o_proj.weight"
                            )
                        ).shape[0]
                    ),
                )
            )
        else:
            a_66 = F.linear(
                a_66,
                self._param(
                    self._join_scope(self._scope_of(node_path_67), "self_attn.o_proj.weight")
                ),
                None,
            )
        x = x + a_66
        node_path_68 = self._join_scope(scope, "n_call_23")
        xnorm_70 = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + float(1e-05))
        xn_69 = xnorm_70 * emitter._param(
            self._join_scope(self._scope_of(node_path_68), "post_attention_layernorm.weight")
        )
        node_path_71 = self._join_scope(scope, "n_op_25")
        if xn_69.numel() == 0:
            pipe_24_72 = xn_69.new_empty(
                (
                    *xn_69.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_71), "mlp.gate.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            pipe_24_72 = F.linear(
                xn_69,
                self._param(self._join_scope(self._scope_of(node_path_71), "mlp.gate.weight")),
                None,
            )
        pipe_26_73 = F.softmax(pipe_24_72, dim=int(-1), dtype=torch.float32)
        topk_scores_74, topk_indices_75 = torch.topk(
            pipe_26_73, int(8), dim=int(-1), largest=True, sorted=True
        )
        m_76 = torch.zeros_like(xn_69)
        to_78 = int(64)
        from_79 = int(0)
        step_80 = int(1)
        for e_77 in self._for_values(from_value=from_79, to_value=to_78, step_value=step_80):
            scope_81 = self._join_scope(scope, f"mlp.experts.{e_77}")
            hidden_flat_86 = xn_69.reshape(-1, xn_69.shape[-1])
            topk_scores_flat_87 = topk_scores_74.reshape(-1, topk_scores_74.shape[-1])
            topk_indices_flat_88 = topk_indices_75.reshape(-1, topk_indices_75.shape[-1])
            expert_pos_89 = (topk_indices_flat_88 == int(e_77)).nonzero(as_tuple=False)
            token_idx_83 = expert_pos_89[:, 0]
            topk_pos_84 = expert_pos_89[:, 1]
            x_sel_82 = hidden_flat_86[token_idx_83]
            sel_scores_85 = topk_scores_flat_87[token_idx_83, topk_pos_84].to(x_sel_82.dtype)
            out_0_90 = self._block_expert_ffn(x=x_sel_82, scope=scope_81)
            x_upd_91 = out_0_90
            m_76 = m_76
            if token_idx_83.numel() != 0:
                _acc = m_76.reshape(-1, m_76.shape[-1])
                _upd = x_upd_91 * sel_scores_85.unsqueeze(-1).to(x_upd_91.dtype)
                _acc.index_add_(0, token_idx_83, _upd.to(_acc.dtype))
        out_0_92 = x + m_76
        return (out_0_92, new_kv_52)

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
        out_0_93 = self._block_Cache_past_length(cache=past_kv, scope=scope)
        kwarg_1_94 = out_0_93
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
            full_pos_97 = attn_mask.to(torch.long).cumsum(dim=-1) - 1
            full_pos_97 = full_pos_97.masked_fill(attn_mask == 0, 0)
            pos_ids_95 = full_pos_97[:, -input_ids.shape[1] :]
        else:
            pos_offset_96 = int(kwarg_1_94)
            if pos_offset_96 < 0:
                raise ValueError("position_ids.past_length must resolve to non-negative int")
            pos_ids_95 = torch.arange(
                pos_offset_96,
                pos_offset_96 + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
        node_path_98 = self._join_scope(scope, "n_op_4")
        x_99 = F.embedding(
            input_ids,
            emitter._param(
                self._join_scope(self._scope_of(node_path_98), "model.embed_tokens.weight")
            ),
        )
        new_kv_100 = None
        if use_cache:
            new_kv_100 = []
        if not (use_cache):
            new_kv_100 = None
        to_102 = int(16)
        from_103 = int(0)
        step_104 = int(1)
        for i_101 in self._for_values(from_value=from_103, to_value=to_102, step_value=step_104):
            scope_105 = self._join_scope(scope, f"model.layers.{i_101}")
            if past_kv is None:
                past_i_106 = None
            else:
                try:
                    past_i_106 = past_kv[int(i_101)]
                except (IndexError, KeyError, TypeError):
                    past_i_106 = None
            out_0_107, new_kv_108 = self._block_olmoe_decoder_block(
                x=x_99, pos_ids=pos_ids_95, attn_mask=attn_mask, past_kv=past_i_106, scope=scope_105
            )
            x_99 = out_0_107
            new_i_109 = new_kv_108
            if new_kv_100 is None:
                new_kv_100 = None
            else:
                new_kv_100 = list(new_kv_100)
                new_kv_100.append(new_i_109)
        node_path_110 = self._join_scope(scope, "n_call_11")
        xnorm_111 = x_99 * torch.rsqrt(torch.mean(x_99 * x_99, dim=-1, keepdim=True) + float(1e-05))
        x_99 = xnorm_111 * emitter._param(
            self._join_scope(self._scope_of(node_path_110), "model.norm.weight")
        )
        node_path_112 = self._join_scope(scope, "n_op_12")
        if x_99.numel() == 0:
            logits_113 = x_99.new_empty(
                (
                    *x_99.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_112), "lm_head.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            logits_113 = F.linear(
                x_99,
                self._param(self._join_scope(self._scope_of(node_path_112), "lm_head.weight")),
                None,
            )
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_113
        outputs["new_kv"] = new_kv_100
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
