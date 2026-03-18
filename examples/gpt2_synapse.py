from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class GPT2Synapse(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int] = {}
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "GPT2Synapse":
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

    def _block_gpt2_block(self, x, scope: str) -> tuple[Any, ...]:
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        node_path_1 = self._join_scope(scope, "ln_1")
        x1_2 = F.layer_norm(
            x,
            (x.shape[-1],),
            weight=emitter._param(self._join_scope(node_path_1, "weight")),
            bias=emitter._param(self._join_scope(node_path_1, "bias")),
            eps=float(1e-05),
        )
        scope_3 = self._join_scope(scope, "attn")
        node_path_4 = self._join_scope(scope_3, "c_attn")
        qkv_5 = torch.matmul(x1_2, self._param(self._join_scope(node_path_4, "weight")))
        if self._state.get(self._join_scope(node_path_4, "bias")) is not None:
            qkv_5 = qkv_5 + self._state.get(self._join_scope(node_path_4, "bias"))
        split_6 = torch.chunk(qkv_5, int(3), dim=-1)
        q_lin_7 = split_6[0]
        k_lin_8 = split_6[1]
        v_lin_9 = split_6[2]
        q_10 = q_lin_7.view(q_lin_7.shape[0], q_lin_7.shape[1], int(12), int(64)).transpose(1, 2)
        k_11 = k_lin_8.view(k_lin_8.shape[0], k_lin_8.shape[1], int(12), int(64)).transpose(1, 2)
        v_12 = v_lin_9.view(v_lin_9.shape[0], v_lin_9.shape[1], int(12), int(64)).transpose(1, 2)
        ctx_heads_13 = F.scaled_dot_product_attention(
            q_10,
            k_11,
            v_12,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(q_10.shape[-2] > 1 and None is None),
            scale=None,
        )
        ctx_14 = (
            ctx_heads_13.transpose(1, 2)
            .contiguous()
            .view(
                ctx_heads_13.shape[0],
                ctx_heads_13.shape[2],
                ctx_heads_13.shape[1] * ctx_heads_13.shape[3],
            )
        )
        scope_15 = self._join_scope(scope, "attn")
        node_path_16 = self._join_scope(scope_15, "c_proj")
        a_17 = torch.matmul(ctx_14, self._param(self._join_scope(node_path_16, "weight")))
        if self._state.get(self._join_scope(node_path_16, "bias")) is not None:
            a_17 = a_17 + self._state.get(self._join_scope(node_path_16, "bias"))
        x2_18 = x + a_17
        node_path_19 = self._join_scope(scope, "ln_2")
        x3_20 = F.layer_norm(
            x2_18,
            (x2_18.shape[-1],),
            weight=emitter._param(self._join_scope(node_path_19, "weight")),
            bias=emitter._param(self._join_scope(node_path_19, "bias")),
            eps=float(1e-05),
        )
        scope_21 = self._join_scope(scope, "mlp")
        node_path_22 = self._join_scope(scope_21, "c_fc")
        m1_23 = torch.matmul(x3_20, self._param(self._join_scope(node_path_22, "weight")))
        if self._state.get(self._join_scope(node_path_22, "bias")) is not None:
            m1_23 = m1_23 + self._state.get(self._join_scope(node_path_22, "bias"))
        m2_24 = (
            0.5
            * m1_23
            * (1.0 + torch.tanh(0.7978845608028654 * (m1_23 + 0.044715 * m1_23 * m1_23 * m1_23)))
        )
        scope_25 = self._join_scope(scope, "mlp")
        node_path_26 = self._join_scope(scope_25, "c_proj")
        m3_27 = torch.matmul(m2_24, self._param(self._join_scope(node_path_26, "weight")))
        if self._state.get(self._join_scope(node_path_26, "bias")) is not None:
            m3_27 = m3_27 + self._state.get(self._join_scope(node_path_26, "bias"))
        y_28 = x2_18 + m3_27
        return y_28

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        if input_ids is not None:
            inputs = {"input_ids": input_ids, **inputs}
        env: dict[str, Any] = dict(inputs)
        scope = ""
        emitter = self
        input_ids = self._safe_get(env, "input_ids")
        node_path_29 = self._join_scope(scope, "wte")
        tok_30 = F.embedding(input_ids, emitter._param(self._join_scope(node_path_29, "weight")))
        pos_ids_31 = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        node_path_32 = self._join_scope(scope, "wpe")
        pos_33 = F.embedding(pos_ids_31, emitter._param(self._join_scope(node_path_32, "weight")))
        x_34 = tok_30 + pos_33
        for i in range(int(12)):
            scope_35 = self._join_scope(scope, f"h.{i}")
            y_36 = self._block_gpt2_block(x=x_34, scope=scope_35)
            x_34 = y_36
        node_path_37 = self._join_scope(scope, "ln_f")
        h_last_38 = F.layer_norm(
            x_34,
            (x_34.shape[-1],),
            weight=emitter._param(self._join_scope(node_path_37, "weight")),
            bias=emitter._param(self._join_scope(node_path_37, "bias")),
            eps=float(1e-05),
        )
        logits_39 = F.linear(h_last_38, self._param("wte.weight"), None)
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_39
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
