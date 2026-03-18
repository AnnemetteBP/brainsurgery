from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class OLMoE1B7B0924Synapse(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int] = {}
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
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        env["past_kv"] = past_kv
        env["use_cache"] = use_cache
        node_path_1 = self._join_scope(scope, "input_layernorm")
        xnorm_3 = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + float(1e-05))
        x_norm_2 = xnorm_3 * emitter._param(self._join_scope(node_path_1, "weight"))
        scope_4 = self._join_scope(scope, "self_attn")
        node_path_5 = self._join_scope(scope_4, "q_proj")
        q_lin_6 = F.linear(x_norm_2, self._param(self._join_scope(node_path_5, "weight")), None)
        scope_7 = self._join_scope(scope, "self_attn")
        node_path_8 = self._join_scope(scope_7, "k_proj")
        k_lin_9 = F.linear(x_norm_2, self._param(self._join_scope(node_path_8, "weight")), None)
        scope_10 = self._join_scope(scope, "self_attn")
        node_path_11 = self._join_scope(scope_10, "v_proj")
        v_lin_12 = F.linear(x_norm_2, self._param(self._join_scope(node_path_11, "weight")), None)
        scope_13 = self._join_scope(scope, "self_attn")
        node_path_14 = self._join_scope(scope_13, "q_norm")
        xnorm_16 = q_lin_6 * torch.rsqrt(
            torch.mean(q_lin_6 * q_lin_6, dim=-1, keepdim=True) + float(1e-05)
        )
        qn_lin_15 = xnorm_16 * emitter._param(self._join_scope(node_path_14, "weight"))
        scope_17 = self._join_scope(scope, "self_attn")
        node_path_18 = self._join_scope(scope_17, "k_norm")
        xnorm_20 = k_lin_9 * torch.rsqrt(
            torch.mean(k_lin_9 * k_lin_9, dim=-1, keepdim=True) + float(1e-05)
        )
        kn_lin_19 = xnorm_20 * emitter._param(self._join_scope(node_path_18, "weight"))
        q_21 = qn_lin_15.view(qn_lin_15.shape[0], qn_lin_15.shape[1], int(16), int(128)).transpose(
            1, 2
        )
        k_22 = kn_lin_19.view(kn_lin_19.shape[0], kn_lin_19.shape[1], int(16), int(128)).transpose(
            1, 2
        )
        v_23 = v_lin_12.view(v_lin_12.shape[0], v_lin_12.shape[1], int(16), int(128)).transpose(
            1, 2
        )
        if past_kv is None:
            past_len_24 = 0
        else:
            past_len_24 = int(past_kv[0].shape[-2])
        half_27 = q_21.shape[-1] // 2
        inv_freq_28 = 1.0 / (
            float(10000.0)
            ** (torch.arange(0, half_27, device=q_21.device, dtype=q_21.dtype) / float(half_27))
        )
        pos_29 = torch.arange(
            int(past_len_24),
            int(past_len_24) + q_21.shape[-2],
            device=q_21.device,
            dtype=q_21.dtype,
        )
        ang_30 = torch.einsum("t,d->td", pos_29, inv_freq_28)
        cos_31 = torch.cos(ang_30)[None, None, :, :]
        sin_32 = torch.sin(ang_30)[None, None, :, :]
        q1_33 = q_21[..., :half_27]
        q2_34 = q_21[..., half_27 : 2 * half_27]
        k1_35 = k_22[..., :half_27]
        k2_36 = k_22[..., half_27 : 2 * half_27]
        qr_25 = torch.cat(
            [q1_33 * cos_31 - q2_34 * sin_32, q1_33 * sin_32 + q2_34 * cos_31], dim=-1
        )
        kr_26 = torch.cat(
            [k1_35 * cos_31 - k2_36 * sin_32, k1_35 * sin_32 + k2_36 * cos_31], dim=-1
        )
        if past_kv is None:
            k_all_37 = kr_26
            v_all_38 = v_23
        else:
            k_all_37 = torch.cat([past_kv[0], kr_26], dim=-2)
            v_all_38 = torch.cat([past_kv[1], v_23], dim=-2)
        present_kv_39 = (k_all_37, v_all_38)
        k_coalesced_40 = k_all_37 if ("k_all_37" in locals() and k_all_37 is not None) else kr_26
        v_coalesced_41 = v_all_38 if ("v_all_38" in locals() and v_all_38 is not None) else v_23
        n_rep_43 = int((int(16) // int(16)))
        if n_rep_43 == 1:
            k_ctx_42 = k_coalesced_40
        else:
            k_ctx_42 = (
                k_coalesced_40[:, :, None, :, :]
                .expand(
                    k_coalesced_40.shape[0],
                    k_coalesced_40.shape[1],
                    n_rep_43,
                    k_coalesced_40.shape[2],
                    k_coalesced_40.shape[3],
                )
                .reshape(
                    k_coalesced_40.shape[0],
                    k_coalesced_40.shape[1] * n_rep_43,
                    k_coalesced_40.shape[2],
                    k_coalesced_40.shape[3],
                )
            )
        n_rep_45 = int((int(16) // int(16)))
        if n_rep_45 == 1:
            v_ctx_44 = v_coalesced_41
        else:
            v_ctx_44 = (
                v_coalesced_41[:, :, None, :, :]
                .expand(
                    v_coalesced_41.shape[0],
                    v_coalesced_41.shape[1],
                    n_rep_45,
                    v_coalesced_41.shape[2],
                    v_coalesced_41.shape[3],
                )
                .reshape(
                    v_coalesced_41.shape[0],
                    v_coalesced_41.shape[1] * n_rep_45,
                    v_coalesced_41.shape[2],
                    v_coalesced_41.shape[3],
                )
            )
        mask_46 = None
        if not use_cache:
            q_len_47 = qr_25.shape[-2]
            k_len_48 = k_ctx_42.shape[-2]
            i_idx_49 = torch.arange(q_len_47, device=qr_25.device).unsqueeze(1)
            j_idx_50 = torch.arange(k_len_48, device=qr_25.device).unsqueeze(0)
            keep_51 = j_idx_50 <= i_idx_49
            window_53 = int(4096)
            if window_53 >= k_len_48 and q_len_47 == k_len_48:
                mask_46 = None
            else:
                keep_51 = keep_51 & (j_idx_50 >= (i_idx_49 - window_53 + 1))
                mask_val_52 = torch.finfo(qr_25.dtype).min
                mask_46 = torch.where(
                    keep_51,
                    torch.zeros((), dtype=qr_25.dtype, device=qr_25.device),
                    torch.full((), mask_val_52, dtype=qr_25.dtype, device=qr_25.device),
                ).view(1, 1, q_len_47, k_len_48)
        ctx_heads_54 = F.scaled_dot_product_attention(
            qr_25,
            k_ctx_42,
            v_ctx_44,
            attn_mask=mask_46,
            dropout_p=0.0,
            is_causal=(qr_25.shape[-2] > 1 and mask_46 is None),
            scale=0.08838834764831845,
        )
        ctx_55 = (
            ctx_heads_54.transpose(1, 2)
            .contiguous()
            .view(
                ctx_heads_54.shape[0],
                ctx_heads_54.shape[2],
                ctx_heads_54.shape[1] * ctx_heads_54.shape[3],
            )
        )
        scope_56 = self._join_scope(scope, "self_attn")
        node_path_57 = self._join_scope(scope_56, "o_proj")
        a_58 = F.linear(ctx_55, self._param(self._join_scope(node_path_57, "weight")), None)
        x2_59 = x + a_58
        node_path_60 = self._join_scope(scope, "post_attention_layernorm")
        xnorm_62 = x2_59 * torch.rsqrt(
            torch.mean(x2_59 * x2_59, dim=-1, keepdim=True) + float(1e-05)
        )
        x3_61 = xnorm_62 * emitter._param(self._join_scope(node_path_60, "weight"))
        scope_63 = self._join_scope(scope, "mlp")
        node_path_64 = self._join_scope(scope_63, "gate")
        router_logits_65 = F.linear(
            x3_61, self._param(self._join_scope(node_path_64, "weight")), None
        )
        router_probs_66 = F.softmax(router_logits_65, dim=int(-1), dtype=torch.float32)
        topk_scores_67, topk_indices_68 = torch.topk(
            router_probs_66, int(8), dim=int(-1), largest=True, sorted=True
        )
        m_69 = torch.zeros_like(x3_61)
        scope_70 = self._join_scope(scope, "mlp")
        for e in range(int(64)):
            scope_71 = self._join_scope(scope_70, f"experts.{e}")
            hidden_flat_76 = x3_61.reshape(-1, x3_61.shape[-1])
            topk_scores_flat_77 = topk_scores_67.reshape(-1, topk_scores_67.shape[-1])
            topk_indices_flat_78 = topk_indices_68.reshape(-1, topk_indices_68.shape[-1])
            expert_pos_79 = (topk_indices_flat_78 == int(e)).nonzero(as_tuple=False)
            token_idx_73 = expert_pos_79[:, 0]
            topk_pos_74 = expert_pos_79[:, 1]
            x_sel_72 = hidden_flat_76[token_idx_73]
            sel_scores_75 = topk_scores_flat_77[token_idx_73, topk_pos_74].to(x_sel_72.dtype)
            experts_base_81 = emitter._join_scope(scope_71, "")
            experts_parent_82 = experts_base_81.rsplit(".", 1)[0] if "." in experts_base_81 else ""
            packed_gate_up_83 = emitter._join_scope(experts_base_81, "gate_up_proj")
            packed_down_84 = self._join_scope(experts_base_81, "down_proj")
            packed_gate_up_parent_85 = (
                emitter._join_scope(experts_parent_82, "gate_up_proj") if experts_parent_82 else ""
            )
            packed_down_parent_86 = (
                emitter._join_scope(experts_parent_82, "down_proj") if experts_parent_82 else ""
            )
            packed_gate_up_key_87 = (
                packed_gate_up_83
                if packed_gate_up_83 in emitter._state
                else (
                    packed_gate_up_parent_85 if packed_gate_up_parent_85 in emitter._state else None
                )
            )
            packed_down_key_88 = (
                packed_down_84
                if packed_down_84 in emitter._state
                else (packed_down_parent_86 if packed_down_parent_86 in emitter._state else None)
            )
            if x_sel_72.numel() == 0:
                if packed_down_key_88 is not None:
                    _down_shape = self._param(packed_down_key_88).shape
                    x_upd_80 = x_sel_72.new_zeros((0, int(_down_shape[-2])))
                else:
                    _dw_indexed = emitter._join_scope(experts_base_81, f"{int(e)}.down_proj.weight")
                    _dw_scoped = emitter._join_scope(experts_base_81, "down_proj.weight")
                    _dw_candidates = [_dw_indexed, _dw_scoped]
                    if experts_parent_82:
                        _dw_candidates.extend(
                            [
                                emitter._join_scope(
                                    experts_parent_82, f"{int(e)}.down_proj.weight"
                                ),
                                emitter._join_scope(experts_parent_82, "down_proj.weight"),
                            ]
                        )
                    _dw = None
                    for _p in _dw_candidates:
                        if _p in self._state:
                            _dw = self._state[_p]
                            break
                    if _dw is None:
                        raise KeyError(_dw_candidates[0])
                    x_upd_80 = x_sel_72.new_zeros((0, int(_dw.shape[-2])))
            else:
                if packed_gate_up_key_87 is not None and packed_down_key_88 is not None:
                    packed_gate_up_w_89 = self._param(packed_gate_up_key_87)
                    if packed_gate_up_w_89.ndim == 3:
                        packed_gate_up_w_89 = packed_gate_up_w_89[int(e)]
                    gate_91, up_92 = F.linear(x_sel_72, packed_gate_up_w_89, None).chunk(2, dim=-1)
                    down_w_90 = self._param(packed_down_key_88)
                    if down_w_90.ndim == 3:
                        down_w_90 = down_w_90[int(e)]
                else:
                    _gw_indexed = emitter._join_scope(experts_base_81, f"{int(e)}.gate_proj.weight")
                    _uw_indexed = emitter._join_scope(experts_base_81, f"{int(e)}.up_proj.weight")
                    _dw_indexed = emitter._join_scope(experts_base_81, f"{int(e)}.down_proj.weight")
                    _gw_scoped = emitter._join_scope(experts_base_81, "gate_proj.weight")
                    _uw_scoped = emitter._join_scope(experts_base_81, "up_proj.weight")
                    _dw_scoped = emitter._join_scope(experts_base_81, "down_proj.weight")
                    _gw_candidates = [_gw_indexed, _gw_scoped]
                    _uw_candidates = [_uw_indexed, _uw_scoped]
                    _dw_candidates = [_dw_indexed, _dw_scoped]
                    if experts_parent_82:
                        _gw_candidates.extend(
                            [
                                emitter._join_scope(
                                    experts_parent_82, f"{int(e)}.gate_proj.weight"
                                ),
                                emitter._join_scope(experts_parent_82, "gate_proj.weight"),
                            ]
                        )
                        _uw_candidates.extend(
                            [
                                emitter._join_scope(experts_parent_82, f"{int(e)}.up_proj.weight"),
                                emitter._join_scope(experts_parent_82, "up_proj.weight"),
                            ]
                        )
                        _dw_candidates.extend(
                            [
                                emitter._join_scope(
                                    experts_parent_82, f"{int(e)}.down_proj.weight"
                                ),
                                emitter._join_scope(experts_parent_82, "down_proj.weight"),
                            ]
                        )
                    _gw = None
                    _uw = None
                    down_w_90 = None
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
                            down_w_90 = self._state[_p]
                            break
                    if _gw is None:
                        raise KeyError(_gw_candidates[0])
                    if _uw is None:
                        raise KeyError(_uw_candidates[0])
                    if down_w_90 is None:
                        raise KeyError(_dw_candidates[0])
                    gate_91 = F.linear(x_sel_72, _gw, None)
                    up_92 = F.linear(x_sel_72, _uw, None)
                hidden_93 = F.silu(gate_91) * up_92
                x_upd_80 = F.linear(hidden_93, down_w_90, None)
            m_69 = m_69
            if token_idx_73.numel() != 0:
                _acc = m_69.reshape(-1, m_69.shape[-1])
                _upd = x_upd_80 * sel_scores_75.unsqueeze(-1).to(x_upd_80.dtype)
                _acc.index_add_(0, token_idx_73, _upd.to(_acc.dtype))
        y_94 = x2_59 + m_69
        return (y_94, present_kv_39)

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        if input_ids is not None:
            inputs = {"input_ids": input_ids, **inputs}
        env: dict[str, Any] = dict(inputs)
        scope = ""
        emitter = self
        input_ids = self._safe_get(env, "input_ids")
        past_key_values = env.get("past_key_values")
        use_cache = env.get("use_cache")
        node_path_95 = self._join_scope(scope, "embed_tokens")
        x_96 = F.embedding(input_ids, emitter._param(self._join_scope(node_path_95, "weight")))
        present_key_values_97 = []
        for i in range(int(16)):
            scope_98 = self._join_scope(scope, f"layers.{i}")
            past_i_99 = None if past_key_values is None else past_key_values[int(i)]
            y_100, present_kv_101 = self._block_olmoe_decoder_block(
                x=x_96, past_kv=past_i_99, use_cache=use_cache, scope=scope_98
            )
            x_96 = y_100
            present_i_102 = present_kv_101
            if use_cache:
                present_key_values_97 = list(present_key_values_97)
                present_key_values_97.append(present_i_102)
        node_path_103 = self._join_scope(scope, "norm")
        xnorm_105 = x_96 * torch.rsqrt(torch.mean(x_96 * x_96, dim=-1, keepdim=True) + float(1e-05))
        h_last_104 = xnorm_105 * emitter._param(self._join_scope(node_path_103, "weight"))
        logits_106 = F.linear(h_last_104, self._param("lm_head.weight"), None)
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_106
        outputs["past_key_values"] = present_key_values_97
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
