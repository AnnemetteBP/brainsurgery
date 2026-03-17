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

    def _block_olmoe_decoder_block(self, x, scope: str) -> tuple[Any, ...]:
        env: dict[str, Any] = {}
        env["x"] = x
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
        half_22 = q_17.shape[-1] // 2
        inv_freq_23 = 1.0 / (
            float(10000.0)
            ** (torch.arange(0, half_22, device=q_17.device, dtype=q_17.dtype) / float(half_22))
        )
        pos_24 = torch.arange(q_17.shape[-2], device=q_17.device, dtype=q_17.dtype)
        ang_25 = torch.einsum("t,d->td", pos_24, inv_freq_23)
        cos_26 = torch.cos(ang_25)[None, None, :, :]
        sin_27 = torch.sin(ang_25)[None, None, :, :]
        q1_28 = q_17[..., :half_22]
        q2_29 = q_17[..., half_22 : 2 * half_22]
        k1_30 = k_18[..., :half_22]
        k2_31 = k_18[..., half_22 : 2 * half_22]
        qr_20 = torch.cat(
            [q1_28 * cos_26 - q2_29 * sin_27, q1_28 * sin_27 + q2_29 * cos_26], dim=-1
        )
        kr_21 = torch.cat(
            [k1_30 * cos_26 - k2_31 * sin_27, k1_30 * sin_27 + k2_31 * cos_26], dim=-1
        )
        n_rep_33 = int((int(16) // int(16)))
        if n_rep_33 == 1:
            k_ctx_32 = kr_21
        else:
            k_ctx_32 = (
                kr_21[:, :, None, :, :]
                .expand(kr_21.shape[0], kr_21.shape[1], n_rep_33, kr_21.shape[2], kr_21.shape[3])
                .reshape(kr_21.shape[0], kr_21.shape[1] * n_rep_33, kr_21.shape[2], kr_21.shape[3])
            )
        n_rep_35 = int((int(16) // int(16)))
        if n_rep_35 == 1:
            v_ctx_34 = v_19
        else:
            v_ctx_34 = (
                v_19[:, :, None, :, :]
                .expand(v_19.shape[0], v_19.shape[1], n_rep_35, v_19.shape[2], v_19.shape[3])
                .reshape(v_19.shape[0], v_19.shape[1] * n_rep_35, v_19.shape[2], v_19.shape[3])
            )
        q_len_37 = qr_20.shape[-2]
        k_len_38 = k_ctx_32.shape[-2]
        i_idx_39 = torch.arange(q_len_37, device=qr_20.device).unsqueeze(1)
        j_idx_40 = torch.arange(k_len_38, device=qr_20.device).unsqueeze(0)
        keep_41 = j_idx_40 <= i_idx_39
        window_43 = int(4096)
        if window_43 >= k_len_38 and q_len_37 == k_len_38:
            mask_36 = None
        else:
            keep_41 = keep_41 & (j_idx_40 >= (i_idx_39 - window_43 + 1))
            mask_val_42 = torch.finfo(qr_20.dtype).min
            mask_36 = torch.where(
                keep_41,
                torch.zeros((), dtype=qr_20.dtype, device=qr_20.device),
                torch.full((), mask_val_42, dtype=qr_20.dtype, device=qr_20.device),
            ).view(1, 1, q_len_37, k_len_38)
        ctx_heads_44 = F.scaled_dot_product_attention(
            qr_20,
            k_ctx_32,
            v_ctx_34,
            attn_mask=mask_36,
            dropout_p=0.0,
            is_causal=(qr_20.shape[-2] > 1 and mask_36 is None),
            scale=0.08838834764831845,
        )
        ctx_45 = (
            ctx_heads_44.transpose(1, 2)
            .contiguous()
            .view(
                ctx_heads_44.shape[0],
                ctx_heads_44.shape[2],
                ctx_heads_44.shape[1] * ctx_heads_44.shape[3],
            )
        )
        node_path_46 = self._join_scope(scope_4, "o_proj")
        a_47 = F.linear(ctx_45, self._param(self._join_scope(node_path_46, "weight")), None)
        x2_48 = x + a_47
        node_path_49 = self._join_scope(scope, "post_attention_layernorm")
        xnorm_51 = x2_48 * torch.rsqrt(
            torch.mean(x2_48 * x2_48, dim=-1, keepdim=True) + float(1e-05)
        )
        x3_50 = xnorm_51 * self._param(self._join_scope(node_path_49, "weight"))
        scope_52 = self._join_scope(scope, "mlp")
        node_path_53 = self._join_scope(scope_52, "gate")
        hidden_flat_54 = x3_50.reshape(-1, x3_50.shape[-1])
        router_logits_55 = F.linear(
            hidden_flat_54, self._param(self._join_scope(node_path_53, "weight")), None
        )
        if int(64) != router_logits_55.shape[-1]:
            raise ValueError("moe_router_topk num_experts mismatch")
        router_probs_56 = F.softmax(router_logits_55, dim=-1, dtype=torch.float32)
        topk_scores_57, topk_indices_58 = torch.topk(router_probs_56, int(8), dim=-1)
        topk_scores_57 = topk_scores_57.to(router_probs_56.dtype)
        hidden_flat_61 = x3_50.reshape(-1, x3_50.shape[-1])
        final_hidden_62 = torch.zeros_like(hidden_flat_61)
        experts_base_60 = self._join_scope(scope_52, "experts")
        packed_gate_up_71 = self._join_scope(experts_base_60, "gate_up_proj")
        packed_down_72 = self._join_scope(experts_base_60, "down_proj")
        for expert_idx_63 in range(int(64)):
            expert_pos_64 = (topk_indices_58 == expert_idx_63).nonzero(as_tuple=False)
            if expert_pos_64.numel() == 0:
                continue
            token_idx_65 = expert_pos_64[:, 0]
            topk_pos_66 = expert_pos_64[:, 1]
            current_state_67 = hidden_flat_61[token_idx_65]
            if packed_gate_up_71 in self._state and packed_down_72 in self._state:
                packed_gate_up_w_73 = self._param(packed_gate_up_71)
                if packed_gate_up_w_73.ndim == 3:
                    packed_gate_up_w_73 = packed_gate_up_w_73[expert_idx_63]
                gate_69, up_70 = F.linear(current_state_67, packed_gate_up_w_73, None).chunk(
                    2, dim=-1
                )
                packed_down_w_74 = self._param(packed_down_72)
                if packed_down_w_74.ndim == 3:
                    packed_down_w_74 = packed_down_w_74[expert_idx_63]
                down_w_77 = packed_down_w_74
                current_hidden_68 = None
            else:
                gate_w_75 = self._param(
                    self._join_scope(experts_base_60, f"{expert_idx_63}.gate_proj.weight")
                )
                up_w_76 = self._param(
                    self._join_scope(experts_base_60, f"{expert_idx_63}.up_proj.weight")
                )
                down_w_77 = self._param(
                    self._join_scope(experts_base_60, f"{expert_idx_63}.down_proj.weight")
                )
                gate_69 = F.linear(current_state_67, gate_w_75, None)
                up_70 = F.linear(current_state_67, up_w_76, None)
            current_hidden_68 = F.silu(gate_69) * up_70
            current_hidden_68 = F.linear(current_hidden_68, down_w_77, None)
            current_hidden_68 = current_hidden_68 * topk_scores_57[
                token_idx_65, topk_pos_66
            ].unsqueeze(-1).to(current_hidden_68.dtype)
            final_hidden_62.index_add_(0, token_idx_65, current_hidden_68.to(final_hidden_62.dtype))
        m_59 = final_hidden_62.reshape_as(x3_50)
        y_78 = x2_48 + m_59
        return y_78

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        if input_ids is not None:
            inputs = {"input_ids": input_ids, **inputs}
        env: dict[str, Any] = dict(inputs)
        scope = ""
        input_ids = self._safe_get(env, "input_ids")
        node_path_79 = self._join_scope(scope, "embed_tokens")
        x_80 = F.embedding(input_ids, self._param(self._join_scope(node_path_79, "weight")))
        for i in range(int(16)):
            scope_81 = self._join_scope(scope, f"layers.{i}")
            y_82 = self._block_olmoe_decoder_block(x=x_80, scope=scope_81)
            x_80 = y_82
        node_path_83 = self._join_scope(scope, "norm")
        xnorm_85 = x_80 * torch.rsqrt(torch.mean(x_80 * x_80, dim=-1, keepdim=True) + float(1e-05))
        h_last_84 = xnorm_85 * self._param(self._join_scope(node_path_83, "weight"))
        logits_86 = F.linear(h_last_84, self._param("lm_head.weight"), None)
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_86
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
