from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class OLMoE1B7B0924Synapse(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int] = {
            "D": 2048,
            "V": 50304,
            "L": 16,
            "H": 16,
            "KvH": 16,
            "Hd": 128,
            "M": 1024,
            "E": 64,
            "Ept": 8,
            "C": 4096,
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

    def _safe_get(self, env: dict[str, Any], name: str) -> Any:
        if name not in env:
            raise ValueError(f"Missing variable in graph env: {name}")
        return env[name]

    def _block_olmoe_decoder_block(self, x, past_kv, use_cache, scope: str) -> tuple[Any, ...]:
        env: dict[str, Any] = {}
        env["x"] = x
        env["past_kv"] = past_kv
        env["use_cache"] = use_cache
        node_path_1 = self._join_scope(scope, "input_layernorm")
        xnorm_3 = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + float(1e-05))
        x_norm_2 = xnorm_3 * self._param(self._join_scope(node_path_1, "weight"))
        scope_4 = self._join_scope(scope, "self_attn")
        node_path_5 = self._join_scope(scope_4, "q_proj")
        q_lin_6 = F.linear(x_norm_2, self._param(self._join_scope(node_path_5, "weight")), None)
        node_path_7 = self._join_scope(scope_4, "k_proj")
        k_lin_8 = F.linear(x_norm_2, self._param(self._join_scope(node_path_7, "weight")), None)
        node_path_9 = self._join_scope(scope_4, "v_proj")
        v_lin_10 = F.linear(x_norm_2, self._param(self._join_scope(node_path_9, "weight")), None)
        node_path_11 = self._join_scope(scope_4, "q_norm")
        xnorm_13 = q_lin_6 * torch.rsqrt(
            torch.mean(q_lin_6 * q_lin_6, dim=-1, keepdim=True) + float(1e-05)
        )
        qn_lin_12 = xnorm_13 * self._param(self._join_scope(node_path_11, "weight"))
        node_path_14 = self._join_scope(scope_4, "k_norm")
        xnorm_16 = k_lin_8 * torch.rsqrt(
            torch.mean(k_lin_8 * k_lin_8, dim=-1, keepdim=True) + float(1e-05)
        )
        kn_lin_15 = xnorm_16 * self._param(self._join_scope(node_path_14, "weight"))
        q_17 = qn_lin_12.view(qn_lin_12.shape[0], qn_lin_12.shape[1], int(16), int(128)).transpose(
            1, 2
        )
        k_18 = kn_lin_15.view(kn_lin_15.shape[0], kn_lin_15.shape[1], int(16), int(128)).transpose(
            1, 2
        )
        v_19 = v_lin_10.view(v_lin_10.shape[0], v_lin_10.shape[1], int(16), int(128)).transpose(
            1, 2
        )
        if past_kv is None:
            past_len_20 = 0
        else:
            past_len_20 = int(past_kv[0].shape[-2])
        half_23 = q_17.shape[-1] // 2
        inv_freq_24 = 1.0 / (
            float(10000.0)
            ** (torch.arange(0, half_23, device=q_17.device, dtype=q_17.dtype) / float(half_23))
        )
        pos_25 = torch.arange(
            int(past_len_20),
            int(past_len_20) + q_17.shape[-2],
            device=q_17.device,
            dtype=q_17.dtype,
        )
        ang_26 = torch.einsum("t,d->td", pos_25, inv_freq_24)
        cos_27 = torch.cos(ang_26)[None, None, :, :]
        sin_28 = torch.sin(ang_26)[None, None, :, :]
        q1_29 = q_17[..., :half_23]
        q2_30 = q_17[..., half_23 : 2 * half_23]
        k1_31 = k_18[..., :half_23]
        k2_32 = k_18[..., half_23 : 2 * half_23]
        qr_21 = torch.cat(
            [q1_29 * cos_27 - q2_30 * sin_28, q1_29 * sin_28 + q2_30 * cos_27], dim=-1
        )
        kr_22 = torch.cat(
            [k1_31 * cos_27 - k2_32 * sin_28, k1_31 * sin_28 + k2_32 * cos_27], dim=-1
        )
        k_all_33 = None
        v_all_34 = None
        present_kv_35 = None
        if use_cache:
            if past_kv is None:
                k_all_33 = kr_22
                v_all_34 = v_19
            else:
                k_all_33 = torch.cat([past_kv[0], kr_22], dim=-2)
                v_all_34 = torch.cat([past_kv[1], v_19], dim=-2)
            present_kv_35 = (k_all_33, v_all_34)
        k_coalesced_36 = k_all_33 if ("k_all_33" in locals() and k_all_33 is not None) else kr_22
        v_coalesced_37 = v_all_34 if ("v_all_34" in locals() and v_all_34 is not None) else v_19
        n_rep_39 = int((int(16) // int(16)))
        if n_rep_39 == 1:
            k_ctx_38 = k_coalesced_36
        else:
            k_ctx_38 = (
                k_coalesced_36[:, :, None, :, :]
                .expand(
                    k_coalesced_36.shape[0],
                    k_coalesced_36.shape[1],
                    n_rep_39,
                    k_coalesced_36.shape[2],
                    k_coalesced_36.shape[3],
                )
                .reshape(
                    k_coalesced_36.shape[0],
                    k_coalesced_36.shape[1] * n_rep_39,
                    k_coalesced_36.shape[2],
                    k_coalesced_36.shape[3],
                )
            )
        n_rep_41 = int((int(16) // int(16)))
        if n_rep_41 == 1:
            v_ctx_40 = v_coalesced_37
        else:
            v_ctx_40 = (
                v_coalesced_37[:, :, None, :, :]
                .expand(
                    v_coalesced_37.shape[0],
                    v_coalesced_37.shape[1],
                    n_rep_41,
                    v_coalesced_37.shape[2],
                    v_coalesced_37.shape[3],
                )
                .reshape(
                    v_coalesced_37.shape[0],
                    v_coalesced_37.shape[1] * n_rep_41,
                    v_coalesced_37.shape[2],
                    v_coalesced_37.shape[3],
                )
            )
        mask_42 = None
        if not use_cache:
            q_len_43 = qr_21.shape[-2]
            k_len_44 = k_ctx_38.shape[-2]
            i_idx_45 = torch.arange(q_len_43, device=qr_21.device).unsqueeze(1)
            j_idx_46 = torch.arange(k_len_44, device=qr_21.device).unsqueeze(0)
            keep_47 = j_idx_46 <= i_idx_45
            window_49 = int(4096)
            if window_49 >= k_len_44 and q_len_43 == k_len_44:
                mask_42 = None
            else:
                keep_47 = keep_47 & (j_idx_46 >= (i_idx_45 - window_49 + 1))
                mask_val_48 = torch.finfo(qr_21.dtype).min
                mask_42 = torch.where(
                    keep_47,
                    torch.zeros((), dtype=qr_21.dtype, device=qr_21.device),
                    torch.full((), mask_val_48, dtype=qr_21.dtype, device=qr_21.device),
                ).view(1, 1, q_len_43, k_len_44)
        ctx_heads_50 = F.scaled_dot_product_attention(
            qr_21,
            k_ctx_38,
            v_ctx_40,
            attn_mask=mask_42,
            dropout_p=0.0,
            is_causal=(qr_21.shape[-2] > 1 and mask_42 is None),
            scale=0.08838834764831845,
        )
        ctx_51 = (
            ctx_heads_50.transpose(1, 2)
            .contiguous()
            .view(
                ctx_heads_50.shape[0],
                ctx_heads_50.shape[2],
                ctx_heads_50.shape[1] * ctx_heads_50.shape[3],
            )
        )
        node_path_52 = self._join_scope(scope_4, "o_proj")
        a_53 = F.linear(ctx_51, self._param(self._join_scope(node_path_52, "weight")), None)
        x2_54 = x + a_53
        node_path_55 = self._join_scope(scope, "post_attention_layernorm")
        xnorm_57 = x2_54 * torch.rsqrt(
            torch.mean(x2_54 * x2_54, dim=-1, keepdim=True) + float(1e-05)
        )
        x3_56 = xnorm_57 * self._param(self._join_scope(node_path_55, "weight"))
        scope_58 = self._join_scope(scope, "mlp")
        node_path_59 = self._join_scope(scope_58, "gate")
        router_logits_60 = F.linear(
            x3_56, self._param(self._join_scope(node_path_59, "weight")), None
        )
        router_probs_61 = F.softmax(router_logits_60, dim=int(-1), dtype=torch.float32)
        topk_scores_62, topk_indices_63 = torch.topk(
            router_probs_61, int(8), dim=int(-1), largest=True, sorted=True
        )
        m_64 = torch.zeros_like(x3_56)
        for e in range(int(64)):
            scope_65 = self._join_scope(scope_58, f"experts.{e}")
            hidden_flat_70 = x3_56.reshape(-1, x3_56.shape[-1])
            topk_scores_flat_71 = topk_scores_62.reshape(-1, topk_scores_62.shape[-1])
            topk_indices_flat_72 = topk_indices_63.reshape(-1, topk_indices_63.shape[-1])
            expert_pos_73 = (topk_indices_flat_72 == int(e)).nonzero(as_tuple=False)
            token_idx_67 = expert_pos_73[:, 0]
            topk_pos_68 = expert_pos_73[:, 1]
            x_sel_66 = hidden_flat_70[token_idx_67]
            sel_scores_69 = topk_scores_flat_71[token_idx_67, topk_pos_68].to(x_sel_66.dtype)
            experts_base_75 = self._join_scope(scope_65, "")
            experts_parent_76 = experts_base_75.rsplit(".", 1)[0] if "." in experts_base_75 else ""
            packed_gate_up_77 = self._join_scope(experts_base_75, "gate_up_proj")
            packed_down_78 = self._join_scope(experts_base_75, "down_proj")
            packed_gate_up_parent_79 = (
                self._join_scope(experts_parent_76, "gate_up_proj") if experts_parent_76 else ""
            )
            packed_down_parent_80 = (
                self._join_scope(experts_parent_76, "down_proj") if experts_parent_76 else ""
            )
            packed_gate_up_key_81 = (
                packed_gate_up_77
                if packed_gate_up_77 in self._state
                else (packed_gate_up_parent_79 if packed_gate_up_parent_79 in self._state else None)
            )
            packed_down_key_82 = (
                packed_down_78
                if packed_down_78 in self._state
                else (packed_down_parent_80 if packed_down_parent_80 in self._state else None)
            )
            if x_sel_66.numel() == 0:
                if packed_down_key_82 is not None:
                    _down_shape = self._param(packed_down_key_82).shape
                    x_upd_74 = x_sel_66.new_zeros((0, int(_down_shape[-2])))
                else:
                    _dw_indexed = self._join_scope(experts_base_75, f"{int(e)}.down_proj.weight")
                    _dw_scoped = self._join_scope(experts_base_75, "down_proj.weight")
                    _dw_candidates = [_dw_indexed, _dw_scoped]
                    if experts_parent_76:
                        _dw_candidates.extend(
                            [
                                self._join_scope(experts_parent_76, f"{int(e)}.down_proj.weight"),
                                self._join_scope(experts_parent_76, "down_proj.weight"),
                            ]
                        )
                    _dw = None
                    for _p in _dw_candidates:
                        if _p in self._state:
                            _dw = self._state[_p]
                            break
                    if _dw is None:
                        raise KeyError(_dw_candidates[0])
                    x_upd_74 = x_sel_66.new_zeros((0, int(_dw.shape[-2])))
            else:
                if packed_gate_up_key_81 is not None and packed_down_key_82 is not None:
                    packed_gate_up_w_83 = self._param(packed_gate_up_key_81)
                    if packed_gate_up_w_83.ndim == 3:
                        packed_gate_up_w_83 = packed_gate_up_w_83[int(e)]
                    gate_85, up_86 = F.linear(x_sel_66, packed_gate_up_w_83, None).chunk(2, dim=-1)
                    down_w_84 = self._param(packed_down_key_82)
                    if down_w_84.ndim == 3:
                        down_w_84 = down_w_84[int(e)]
                else:
                    _gw_indexed = self._join_scope(experts_base_75, f"{int(e)}.gate_proj.weight")
                    _uw_indexed = self._join_scope(experts_base_75, f"{int(e)}.up_proj.weight")
                    _dw_indexed = self._join_scope(experts_base_75, f"{int(e)}.down_proj.weight")
                    _gw_scoped = self._join_scope(experts_base_75, "gate_proj.weight")
                    _uw_scoped = self._join_scope(experts_base_75, "up_proj.weight")
                    _dw_scoped = self._join_scope(experts_base_75, "down_proj.weight")
                    _gw_candidates = [_gw_indexed, _gw_scoped]
                    _uw_candidates = [_uw_indexed, _uw_scoped]
                    _dw_candidates = [_dw_indexed, _dw_scoped]
                    if experts_parent_76:
                        _gw_candidates.extend(
                            [
                                self._join_scope(experts_parent_76, f"{int(e)}.gate_proj.weight"),
                                self._join_scope(experts_parent_76, "gate_proj.weight"),
                            ]
                        )
                        _uw_candidates.extend(
                            [
                                self._join_scope(experts_parent_76, f"{int(e)}.up_proj.weight"),
                                self._join_scope(experts_parent_76, "up_proj.weight"),
                            ]
                        )
                        _dw_candidates.extend(
                            [
                                self._join_scope(experts_parent_76, f"{int(e)}.down_proj.weight"),
                                self._join_scope(experts_parent_76, "down_proj.weight"),
                            ]
                        )
                    _gw = None
                    _uw = None
                    down_w_84 = None
                    for _p in _gw_candidates:
                        if _p in self._state:
                            _gw = self._state[_p]
                            break
                    for _p in _uw_candidates:
                        if _p in self._state:
                            _uw = self._state[_p]
                            break
                    for _p in _dw_candidates:
                        if _p in self._state:
                            down_w_84 = self._state[_p]
                            break
                    if _gw is None:
                        raise KeyError(_gw_candidates[0])
                    if _uw is None:
                        raise KeyError(_uw_candidates[0])
                    if down_w_84 is None:
                        raise KeyError(_dw_candidates[0])
                    gate_85 = F.linear(x_sel_66, _gw, None)
                    up_86 = F.linear(x_sel_66, _uw, None)
                hidden_87 = F.silu(gate_85) * up_86
                x_upd_74 = F.linear(hidden_87, down_w_84, None)
            m_64 = m_64
            if token_idx_67.numel() != 0:
                _acc = m_64.reshape(-1, m_64.shape[-1])
                _upd = x_upd_74 * sel_scores_69.unsqueeze(-1).to(x_upd_74.dtype)
                _acc.index_add_(0, token_idx_67, _upd.to(_acc.dtype))
        y_88 = x2_54 + m_64
        return (y_88, present_kv_35)

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        if input_ids is not None:
            inputs = {"input_ids": input_ids, **inputs}
        env: dict[str, Any] = dict(inputs)
        scope = ""
        input_ids = self._safe_get(env, "input_ids")
        past_key_values = env.get("past_key_values")
        use_cache = env.get("use_cache")
        node_path_89 = self._join_scope(scope, "embed_tokens")
        x_90 = F.embedding(input_ids, self._param(self._join_scope(node_path_89, "weight")))
        present_key_values_91 = []
        for i in range(int(16)):
            scope_92 = self._join_scope(scope, f"layers.{i}")
            past_i_93 = None if past_key_values is None else past_key_values[int(i)]
            y_94, present_kv_95 = self._block_olmoe_decoder_block(
                x=x_90, past_kv=past_i_93, use_cache=use_cache, scope=scope_92
            )
            x_90 = y_94
            present_i_96 = present_kv_95
            if use_cache:
                present_key_values_91 = list(present_key_values_91)
                present_key_values_91.append(present_i_96)
        node_path_97 = self._join_scope(scope, "norm")
        xnorm_99 = x_90 * torch.rsqrt(torch.mean(x_90 * x_90, dim=-1, keepdim=True) + float(1e-05))
        h_last_98 = xnorm_99 * self._param(self._join_scope(node_path_97, "weight"))
        logits_100 = F.linear(h_last_98, self._param("lm_head.weight"), None)
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_100
        outputs["past_key_values"] = present_key_values_91
        if "logits" in outputs and len(outputs) == 1:
            return outputs["logits"]
        return outputs

    def generate(self, input_ids: torch.Tensor, *, eos_token_id: int, max_len: int) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if input_ids.size(1) >= max_len:
            return input_ids[:, :max_len]

        generated = input_ids
        past_key_values = None
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                while generated.size(1) < max_len and not torch.all(finished):
                    step_input = generated if past_key_values is None else generated[:, -1:]
                    model_out = self.forward(
                        step_input, past_key_values=past_key_values, use_cache=True
                    )
                    if isinstance(model_out, dict):
                        logits = model_out["logits"]
                        if "past_key_values" in model_out:
                            past_key_values = model_out["past_key_values"]
                    else:
                        logits = model_out
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    next_token = torch.where(
                        finished, torch.full_like(next_token, eos_token_id), next_token
                    )
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    finished = torch.logical_or(finished, next_token == eos_token_id)
        finally:
            if was_training:
                self.train()
        return generated
