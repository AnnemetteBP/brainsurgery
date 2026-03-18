from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Gemma3Synapse270M(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int] = {}
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

    def _safe_get(self, env: dict[str, Any], name: str) -> Any:
        if name not in env:
            raise ValueError(f"Missing variable in graph env: {name}")
        return env[name]

    def _block_gemma3_block(self, x, i, scope: str) -> tuple[Any, ...]:
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        env["i"] = i
        node_path_1 = self._join_scope(scope, "input_layernorm")
        xnorm_3 = x.float() * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        x_norm_2 = xnorm_3 * (1.0 + emitter._param(self._join_scope(node_path_1, "weight")).float())
        x_norm_2 = x_norm_2.type_as(x)
        scope_4 = self._join_scope(scope, "self_attn")
        node_path_5 = self._join_scope(scope_4, "q_proj")
        q_lin_6 = F.linear(x_norm_2, self._param(self._join_scope(node_path_5, "weight")), None)
        scope_7 = self._join_scope(scope, "self_attn")
        node_path_8 = self._join_scope(scope_7, "k_proj")
        k_lin_9 = F.linear(x_norm_2, self._param(self._join_scope(node_path_8, "weight")), None)
        scope_10 = self._join_scope(scope, "self_attn")
        node_path_11 = self._join_scope(scope_10, "v_proj")
        v_lin_12 = F.linear(x_norm_2, self._param(self._join_scope(node_path_11, "weight")), None)
        q_13 = q_lin_6.view(q_lin_6.shape[0], q_lin_6.shape[1], int(4), int(256)).transpose(1, 2)
        k_14 = k_lin_9.view(k_lin_9.shape[0], k_lin_9.shape[1], int(1), int(256)).transpose(1, 2)
        v_15 = v_lin_12.view(v_lin_12.shape[0], v_lin_12.shape[1], int(1), int(256)).transpose(1, 2)
        scope_16 = self._join_scope(scope, "self_attn")
        node_path_17 = self._join_scope(scope_16, "q_norm")
        xnorm_19 = q_13.float() * torch.rsqrt(
            torch.mean(q_13.float() * q_13.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        qn_18 = xnorm_19 * (1.0 + emitter._param(self._join_scope(node_path_17, "weight")).float())
        qn_18 = qn_18.type_as(q_13)
        scope_20 = self._join_scope(scope, "self_attn")
        node_path_21 = self._join_scope(scope_20, "k_norm")
        xnorm_23 = k_14.float() * torch.rsqrt(
            torch.mean(k_14.float() * k_14.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        kn_22 = xnorm_23 * (1.0 + emitter._param(self._join_scope(node_path_21, "weight")).float())
        kn_22 = kn_22.type_as(k_14)
        half_26 = qn_18.shape[-1] // 2
        inv_freq_27 = 1.0 / (
            float(10000.0 + (1000000.0 - 10000.0) * (((i + 1) % 6) == 0))
            ** (torch.arange(0, half_26, device=qn_18.device, dtype=qn_18.dtype) / float(half_26))
        )
        pos_28 = torch.arange(
            int(0), int(0) + qn_18.shape[-2], device=qn_18.device, dtype=qn_18.dtype
        )
        ang_29 = torch.einsum("t,d->td", pos_28, inv_freq_27)
        cos_30 = torch.cos(ang_29)[None, None, :, :]
        sin_31 = torch.sin(ang_29)[None, None, :, :]
        q1_32 = qn_18[..., :half_26]
        q2_33 = qn_18[..., half_26 : 2 * half_26]
        k1_34 = kn_22[..., :half_26]
        k2_35 = kn_22[..., half_26 : 2 * half_26]
        qr_24 = torch.cat(
            [q1_32 * cos_30 - q2_33 * sin_31, q1_32 * sin_31 + q2_33 * cos_30], dim=-1
        )
        kr_25 = torch.cat(
            [k1_34 * cos_30 - k2_35 * sin_31, k1_34 * sin_31 + k2_35 * cos_30], dim=-1
        )
        n_rep_37 = int((int(4) // int(1)))
        if n_rep_37 == 1:
            k_ctx_36 = kr_25
        else:
            k_ctx_36 = (
                kr_25[:, :, None, :, :]
                .expand(kr_25.shape[0], kr_25.shape[1], n_rep_37, kr_25.shape[2], kr_25.shape[3])
                .reshape(kr_25.shape[0], kr_25.shape[1] * n_rep_37, kr_25.shape[2], kr_25.shape[3])
            )
        n_rep_39 = int((int(4) // int(1)))
        if n_rep_39 == 1:
            v_ctx_38 = v_15
        else:
            v_ctx_38 = (
                v_15[:, :, None, :, :]
                .expand(v_15.shape[0], v_15.shape[1], n_rep_39, v_15.shape[2], v_15.shape[3])
                .reshape(v_15.shape[0], v_15.shape[1] * n_rep_39, v_15.shape[2], v_15.shape[3])
            )
        q_len_41 = qr_24.shape[-2]
        k_len_42 = k_ctx_36.shape[-2]
        i_idx_43 = torch.arange(q_len_41, device=qr_24.device).unsqueeze(1)
        j_idx_44 = torch.arange(k_len_42, device=qr_24.device).unsqueeze(0)
        keep_45 = j_idx_44 <= i_idx_43
        window_47 = int(512 + (32768 - 512) * (((i + 1) % 6) == 0))
        if window_47 >= k_len_42 and q_len_41 == k_len_42:
            mask_40 = None
        else:
            keep_45 = keep_45 & (j_idx_44 >= (i_idx_43 - window_47 + 1))
            mask_val_46 = torch.finfo(qr_24.dtype).min
            mask_40 = torch.where(
                keep_45,
                torch.zeros((), dtype=qr_24.dtype, device=qr_24.device),
                torch.full((), mask_val_46, dtype=qr_24.dtype, device=qr_24.device),
            ).view(1, 1, q_len_41, k_len_42)
        ctx_heads_48 = F.scaled_dot_product_attention(
            qr_24,
            k_ctx_36,
            v_ctx_38,
            attn_mask=mask_40,
            dropout_p=0.0,
            is_causal=(qr_24.shape[-2] > 1 and mask_40 is None),
            scale=0.0625,
        )
        ctx_49 = (
            ctx_heads_48.transpose(1, 2)
            .contiguous()
            .view(
                ctx_heads_48.shape[0],
                ctx_heads_48.shape[2],
                ctx_heads_48.shape[1] * ctx_heads_48.shape[3],
            )
        )
        scope_50 = self._join_scope(scope, "self_attn")
        node_path_51 = self._join_scope(scope_50, "o_proj")
        attn_raw_52 = F.linear(ctx_49, self._param(self._join_scope(node_path_51, "weight")), None)
        node_path_53 = self._join_scope(scope, "post_attention_layernorm")
        xnorm_55 = attn_raw_52.float() * torch.rsqrt(
            torch.mean(attn_raw_52.float() * attn_raw_52.float(), dim=-1, keepdim=True)
            + float(1e-06)
        )
        attn_54 = xnorm_55 * (
            1.0 + emitter._param(self._join_scope(node_path_53, "weight")).float()
        )
        attn_54 = attn_54.type_as(attn_raw_52)
        x2_56 = x + attn_54
        node_path_57 = self._join_scope(scope, "pre_feedforward_layernorm")
        xnorm_59 = x2_56.float() * torch.rsqrt(
            torch.mean(x2_56.float() * x2_56.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        x3_58 = xnorm_59 * (1.0 + emitter._param(self._join_scope(node_path_57, "weight")).float())
        x3_58 = x3_58.type_as(x2_56)
        scope_60 = self._join_scope(scope, "mlp")
        node_path_61 = self._join_scope(scope_60, "gate_proj")
        g_62 = F.linear(x3_58, self._param(self._join_scope(node_path_61, "weight")), None)
        scope_63 = self._join_scope(scope, "mlp")
        node_path_64 = self._join_scope(scope_63, "up_proj")
        u_65 = F.linear(x3_58, self._param(self._join_scope(node_path_64, "weight")), None)
        g_act_66 = (
            0.5
            * g_62
            * (1.0 + torch.tanh(0.7978845608028654 * (g_62 + 0.044715 * g_62 * g_62 * g_62)))
        )
        gu_67 = g_act_66 * u_65
        scope_68 = self._join_scope(scope, "mlp")
        node_path_69 = self._join_scope(scope_68, "down_proj")
        m_raw_70 = F.linear(gu_67, self._param(self._join_scope(node_path_69, "weight")), None)
        node_path_71 = self._join_scope(scope, "post_feedforward_layernorm")
        xnorm_73 = m_raw_70.float() * torch.rsqrt(
            torch.mean(m_raw_70.float() * m_raw_70.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        m_72 = xnorm_73 * (1.0 + emitter._param(self._join_scope(node_path_71, "weight")).float())
        m_72 = m_72.type_as(m_raw_70)
        y_74 = x2_56 + m_72
        return y_74

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        if input_ids is not None:
            inputs = {"input_ids": input_ids, **inputs}
        env: dict[str, Any] = dict(inputs)
        scope = ""
        emitter = self
        input_ids = self._safe_get(env, "input_ids")
        node_path_75 = self._join_scope(scope, "embed_tokens")
        x_76 = F.embedding(input_ids, emitter._param(self._join_scope(node_path_75, "weight")))
        x_76 = x_76 * torch.tensor(float(25.298221281347036), dtype=x_76.dtype, device=x_76.device)
        for i in range(int(18)):
            scope_77 = self._join_scope(scope, f"layers.{i}")
            y_78 = self._block_gemma3_block(x=x_76, i=i, scope=scope_77)
            x_76 = y_78
        node_path_79 = self._join_scope(scope, "norm")
        xnorm_81 = x_76.float() * torch.rsqrt(
            torch.mean(x_76.float() * x_76.float(), dim=-1, keepdim=True) + float(1e-06)
        )
        h_last_80 = xnorm_81 * (
            1.0 + emitter._param(self._join_scope(node_path_79, "weight")).float()
        )
        h_last_80 = h_last_80.type_as(x_76)
        logits_82 = F.linear(h_last_80, self._param("embed_tokens.weight"), None)
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_82
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
