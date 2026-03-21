from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class GPT2Synapse(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, int | float | bool] = {"T": 1024, "D": 768, "L": 12, "H": 12}
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

    def _block_gpt2_block(self, x, attn_mask, scope: str) -> tuple[Any, ...]:
        emitter = self
        env: dict[str, Any] = {}
        env["x"] = x
        env["attn_mask"] = attn_mask
        node_path_1 = self._join_scope(scope, "n_op_1")
        x1_2 = F.layer_norm(
            x,
            (x.shape[-1],),
            weight=emitter._param(self._join_scope(self._scope_of(node_path_1), "ln_1.weight")),
            bias=emitter._param(self._join_scope(self._scope_of(node_path_1), "ln_1.bias")),
            eps=float(1e-05),
        )
        node_path_3 = self._join_scope(scope, "n_call_3")
        if x1_2.numel() == 0:
            pipe_2_4 = x1_2.new_empty(
                (
                    *x1_2.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_3), "attn.c_attn.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            pipe_2_4 = torch.matmul(
                x1_2,
                self._param(self._join_scope(self._scope_of(node_path_3), "attn.c_attn.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_3), "attn.c_attn.bias"))
                is not None
            ):
                pipe_2_4 = pipe_2_4 + self._state.get(
                    self._join_scope(self._scope_of(node_path_3), "attn.c_attn.bias")
                )
        split_5 = torch.chunk(pipe_2_4, int(3), dim=-1)
        q_lin_6 = split_5[0]
        k_lin_7 = split_5[1]
        v_lin_8 = split_5[2]
        heads_9 = int(12)
        head_dim_10 = None
        if heads_9 is None and head_dim_10 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_9 is None:
            if q_lin_6.shape[-1] % int(head_dim_10) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_9 = q_lin_6.shape[-1] // int(head_dim_10)
        if head_dim_10 is None:
            if q_lin_6.shape[-1] % int(heads_9) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_10 = q_lin_6.shape[-1] // int(heads_9)
        expected_hidden_11 = int(heads_9) * int(head_dim_10)
        if q_lin_6.shape[-1] != expected_hidden_11:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        q_12 = q_lin_6.view(
            q_lin_6.shape[0], q_lin_6.shape[1], int(heads_9), int(head_dim_10)
        ).transpose(1, 2)
        heads_13 = int(12)
        head_dim_14 = None
        if heads_13 is None and head_dim_14 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_13 is None:
            if k_lin_7.shape[-1] % int(head_dim_14) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_13 = k_lin_7.shape[-1] // int(head_dim_14)
        if head_dim_14 is None:
            if k_lin_7.shape[-1] % int(heads_13) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_14 = k_lin_7.shape[-1] // int(heads_13)
        expected_hidden_15 = int(heads_13) * int(head_dim_14)
        if k_lin_7.shape[-1] != expected_hidden_15:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        k_16 = k_lin_7.view(
            k_lin_7.shape[0], k_lin_7.shape[1], int(heads_13), int(head_dim_14)
        ).transpose(1, 2)
        heads_17 = int(12)
        head_dim_18 = None
        if heads_17 is None and head_dim_18 is None:
            raise ValueError("reshape_heads requires heads or head_dim")
        if heads_17 is None:
            if v_lin_8.shape[-1] % int(head_dim_18) != 0:
                raise ValueError("reshape_heads could not infer heads from head_dim")
            heads_17 = v_lin_8.shape[-1] // int(head_dim_18)
        if head_dim_18 is None:
            if v_lin_8.shape[-1] % int(heads_17) != 0:
                raise ValueError("reshape_heads could not infer head_dim from heads")
            head_dim_18 = v_lin_8.shape[-1] // int(heads_17)
        expected_hidden_19 = int(heads_17) * int(head_dim_18)
        if v_lin_8.shape[-1] != expected_hidden_19:
            raise ValueError("reshape_heads heads*head_dim must equal input width")
        v_20 = v_lin_8.view(
            v_lin_8.shape[0], v_lin_8.shape[1], int(heads_17), int(head_dim_18)
        ).transpose(1, 2)
        q_len_22 = q_12.shape[-2]
        k_len_23 = k_16.shape[-2]
        if not hasattr(self, "_causal_mask_cache"):
            self._causal_mask_cache = {}
        window_30 = int(768)
        if attn_mask is None:
            pad_key_31 = None
        else:
            pad_key_31 = (
                int(attn_mask.data_ptr()),
                int(attn_mask.storage_offset()),
                tuple(int(x) for x in attn_mask.shape),
            )
        cache_key_24 = (
            int(q_len_22),
            int(k_len_23),
            window_30,
            q_12.dtype,
            q_12.device,
            pad_key_31,
        )
        cached_mask_25 = self._causal_mask_cache.get(cache_key_24)
        if torch.is_tensor(cached_mask_25):
            mask_21 = cached_mask_25
        else:
            j_idx_27 = torch.arange(k_len_23, device=q_12.device).unsqueeze(0)
            if q_len_22 == 1:
                keep_28 = j_idx_27 >= (k_len_23 - window_30)
            else:
                i_idx_26 = torch.arange(q_len_22, device=q_12.device).unsqueeze(1)
                keep_28 = j_idx_27 <= i_idx_26
                keep_28 = keep_28 & (j_idx_27 >= (i_idx_26 - window_30 + 1))
            if attn_mask is not None:
                if attn_mask.ndim != 2:
                    raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
                if int(attn_mask.shape[-1]) != k_len_23:
                    raise ValueError(
                        "causal_mask.padding_mask width must match key sequence length"
                    )
                pad_keep_32 = attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
                keep_28 = keep_28.unsqueeze(0).unsqueeze(0) & pad_keep_32
            else:
                keep_28 = keep_28.view(1, 1, q_len_22, k_len_23)
            mask_val_29 = torch.finfo(q_12.dtype).min
            mask_21 = torch.where(
                keep_28,
                torch.zeros((), dtype=q_12.dtype, device=q_12.device),
                torch.full((), mask_val_29, dtype=q_12.dtype, device=q_12.device),
            )
            self._causal_mask_cache[cache_key_24] = mask_21
        pipe_9_33 = F.scaled_dot_product_attention(
            q_12,
            k_16,
            v_20,
            attn_mask=mask_21,
            dropout_p=0.0,
            is_causal=(q_12.shape[-2] > 1 and mask_21 is None),
            scale=None,
        )
        pipe_11_34 = (
            pipe_9_33.transpose(1, 2)
            .contiguous()
            .view(pipe_9_33.shape[0], pipe_9_33.shape[2], pipe_9_33.shape[1] * pipe_9_33.shape[3])
        )
        node_path_35 = self._join_scope(scope, "n_call_13")
        if pipe_11_34.numel() == 0:
            out_0_36 = pipe_11_34.new_empty(
                (
                    *pipe_11_34.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_35), "attn.c_proj.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            out_0_36 = torch.matmul(
                pipe_11_34,
                self._param(self._join_scope(self._scope_of(node_path_35), "attn.c_proj.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_35), "attn.c_proj.bias"))
                is not None
            ):
                out_0_36 = out_0_36 + self._state.get(
                    self._join_scope(self._scope_of(node_path_35), "attn.c_proj.bias")
                )
        a_37 = out_0_36
        x = x + a_37
        node_path_38 = self._join_scope(scope, "n_op_16")
        x3_39 = F.layer_norm(
            x,
            (x.shape[-1],),
            weight=emitter._param(self._join_scope(self._scope_of(node_path_38), "ln_2.weight")),
            bias=emitter._param(self._join_scope(self._scope_of(node_path_38), "ln_2.bias")),
            eps=float(1e-05),
        )
        node_path_40 = self._join_scope(scope, "n_call_18")
        if x3_39.numel() == 0:
            pipe_17_41 = x3_39.new_empty(
                (
                    *x3_39.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_40), "mlp.c_fc.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            pipe_17_41 = torch.matmul(
                x3_39,
                self._param(self._join_scope(self._scope_of(node_path_40), "mlp.c_fc.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_40), "mlp.c_fc.bias"))
                is not None
            ):
                pipe_17_41 = pipe_17_41 + self._state.get(
                    self._join_scope(self._scope_of(node_path_40), "mlp.c_fc.bias")
                )
        pipe_19_42 = (
            0.5
            * pipe_17_41
            * (
                1.0
                + torch.tanh(
                    0.7978845608028654
                    * (pipe_17_41 + 0.044715 * pipe_17_41 * pipe_17_41 * pipe_17_41)
                )
            )
        )
        node_path_43 = self._join_scope(scope, "n_call_21")
        if pipe_19_42.numel() == 0:
            out_0_36 = pipe_19_42.new_empty(
                (
                    *pipe_19_42.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_43), "mlp.c_proj.weight")
                        ).shape[-1]
                    ),
                )
            )
        else:
            out_0_36 = torch.matmul(
                pipe_19_42,
                self._param(self._join_scope(self._scope_of(node_path_43), "mlp.c_proj.weight")),
            )
            if (
                self._state.get(self._join_scope(self._scope_of(node_path_43), "mlp.c_proj.bias"))
                is not None
            ):
                out_0_36 = out_0_36 + self._state.get(
                    self._join_scope(self._scope_of(node_path_43), "mlp.c_proj.bias")
                )
        m_44 = out_0_36
        out_0_36 = x + m_44
        return out_0_36

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        input_specs = {"input_ids": {"optional": False}, "attn_mask": {"optional": True}}
        env = self._prepare_env(input_ids, inputs, input_specs)
        scope = ""
        emitter = self
        input_ids = self._safe_get(env, "input_ids")
        attn_mask = env.get("attn_mask")
        node_path_45 = self._join_scope(scope, "n_op_1")
        tok_46 = F.embedding(
            input_ids, emitter._param(self._join_scope(self._scope_of(node_path_45), "wte.weight"))
        )
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
            full_pos_49 = attn_mask.to(torch.long).cumsum(dim=-1) - 1
            full_pos_49 = full_pos_49.masked_fill(attn_mask == 0, 0)
            pipe_2_47 = full_pos_49[:, -input_ids.shape[1] :]
        else:
            pos_offset_48 = int(0)
            if pos_offset_48 < 0:
                raise ValueError("position_ids.past_length must resolve to non-negative int")
            pipe_2_47 = torch.arange(
                pos_offset_48,
                pos_offset_48 + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
        node_path_50 = self._join_scope(scope, "n_op_4")
        pos_51 = F.embedding(
            pipe_2_47, emitter._param(self._join_scope(self._scope_of(node_path_50), "wpe.weight"))
        )
        x_52 = tok_46 + pos_51
        to_54 = int(12)
        from_55 = int(0)
        step_56 = int(1)
        for i_53 in self._for_values(from_value=from_55, to_value=to_54, step_value=step_56):
            scope_57 = self._join_scope(scope, f"h.{i_53}")
            out_0_58 = self._block_gpt2_block(x=x_52, attn_mask=attn_mask, scope=scope_57)
            x_52 = out_0_58
        node_path_59 = self._join_scope(scope, "n_op_9")
        pipe_8_60 = F.layer_norm(
            x_52,
            (x_52.shape[-1],),
            weight=emitter._param(self._join_scope(self._scope_of(node_path_59), "ln_f.weight")),
            bias=emitter._param(self._join_scope(self._scope_of(node_path_59), "ln_f.bias")),
            eps=float(1e-05),
        )
        node_path_61 = self._join_scope(scope, "n_op_10")
        if pipe_8_60.numel() == 0:
            logits_62 = pipe_8_60.new_empty(
                (
                    *pipe_8_60.shape[:-1],
                    int(
                        self._param(
                            self._join_scope(self._scope_of(node_path_61), "wte.weight")
                        ).shape[0]
                    ),
                )
            )
        else:
            logits_62 = F.linear(
                pipe_8_60,
                self._param(self._join_scope(self._scope_of(node_path_61), "wte.weight")),
                None,
            )
        outputs: dict[str, Any] = {}
        outputs["logits"] = logits_62
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
