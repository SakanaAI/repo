# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py
# Copyright 2024 The vLLM team.
# Copyright 2024 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only OLMo2 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from functools import partial
from typing import Optional, Union

import torch
from torch import nn
from transformers import Olmo2Config
import torch.nn.functional as F

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors



class DynamicPeMode:
    N = 'none'
    C = 'continuous'
    MHC = 'multihead_continuous'
    D = 'discrete'
    CWSC = 'continuous_with_scale_control'
    A = 'attention'
    SWIGLU = 'swiglu'
    SWIGLUEX = 'swigluex'
    NOPE = 'nope'
    SWIFLUSCI = 'swiglu_scientific'
    SST = 'swiglu_scientific_tanh'
    SEXPD = 'swigluex_posdim'
    SEXMHPD = 'swigluex_multihead_posdim'
    SEXSIG = 'swigluex_sigmoid'
    SEXMH = 'swigluex_multihead'
    SEXPA = 'swigluex_posavg'
    SEXFEX = 'swigluex_filterex'
    SEXMHFEX = 'swigluex_multihead_filterex'

    @staticmethod
    def verify(x):
        return x in ['none', 'multihead_continuous', 'continuous', 'discrete', 'continuous_with_scale_control', 'attention', 'swiglu', 'swigluex', 'nope', 'swiglu_scientific', 'swiglu_scientific_tanh', 'swigluex_posdim', 'swigluex_filterex', 'swigluex_posavg', 'swigluex_multihead', 'swigluex_sigmoid', 'swigluex_multihead_posdim', 'swigluex_multihead_filterex']

class Olmo2Attention(nn.Module):
    """
    This is the attention block where the output is computed as
    ``Attention(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", dynamic_pe_mode: Optional[DynamicPeMode] = DynamicPeMode.N):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        assert isinstance(self.config, Olmo2Config)

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (self.config.num_key_value_heads
                                   or self.total_num_heads)
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.tp_rank = get_tensor_model_parallel_rank()
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim,
            eps=self.config.rms_norm_eps,
        )
        self.q_norm = RMSNorm(self.config.hidden_size,
                              eps=self.config.rms_norm_eps)

        # Rotary embeddings.
        rope_scaling = vllm_config.model_config.rope_scaling
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,  # type: ignore
            rope_scaling=rope_scaling if isinstance(rope_scaling, dict) and rope_scaling else None
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )

        # Attention output projection.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.dynamic_pe_mode = dynamic_pe_mode
        if self.dynamic_pe_mode in [DynamicPeMode.C, DynamicPeMode.CWSC]:
            self.dynamic_map = RowParallelLinear(
                self.config.hidden_size, 1, bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.dynamic_map",
            )
        elif self.dynamic_pe_mode == DynamicPeMode.MHC:
            self.dynamic_map = RowParallelLinear(
                self.config.hidden_size, self.total_num_heads, bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.dynamic_map",
            )
        elif self.dynamic_pe_mode == DynamicPeMode.SEXMH:
            mid_size = self.config.hidden_size // 8
            self.gate_map = RowParallelLinear(
                self.config.hidden_size, mid_size, bias=False,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.gate_map"
            )
            self.content_map = RowParallelLinear(
                self.config.hidden_size, mid_size, bias=False,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.content_map"
            )
            self.final_map = RowParallelLinear(
                mid_size, self.total_num_heads, bias=False,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.final_map"
            )

    def dynamic_position_encoding_swigluex_multihead(self, hidden_states, position_ids):
        device, dtype = hidden_states.device, hidden_states.dtype
        # [B*N, D] -> [B*N, H]
        pred_indices = self.final_map(F.silu(self.gate_map(hidden_states)[0]) * self.content_map(hidden_states)[0])[0]
        return pred_indices

    def dynamic_position_encoding_continuous(self, hidden_states, position_ids):
        # hidden_states [B, L, D]
        # pe [B, P, D], where P >= L
        device, dtype = hidden_states.device, hidden_states.dtype
        pred_indices = self.dynamic_map(hidden_states)[0].squeeze(-1)
        return pred_indices

    def dynamic_position_encoding_multihead_continuous(self, hidden_states, position_ids):
        # hidden_states [B, L, D]
        # pe [B, P, D], where P >= L
        device, dtype = hidden_states.device, hidden_states.dtype
        pred_indices = self.dynamic_map(hidden_states)[0]
        return pred_indices

    def _apply_qk_norm(self, q: torch.Tensor,
                       k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        if self.dynamic_pe_mode == DynamicPeMode.N:
            # q, k = self.rotary_emb(positions, q, k)
            q, k = self.rotary_emb.forward_native(positions, q, k, is_dynamic_pe=False)
        elif self.dynamic_pe_mode != DynamicPeMode.NOPE:
            if self.dynamic_pe_mode == DynamicPeMode.C:
                positions = self.dynamic_position_encoding_continuous(hidden_states, positions)
            elif self.dynamic_pe_mode == DynamicPeMode.MHC:
                positions = self.dynamic_position_encoding_multihead_continuous(hidden_states, positions)
            elif self.dynamic_pe_mode == DynamicPeMode.SWIGLU:
                positions = self.dynamic_position_encoding_swiglu(hidden_states, positions)
            elif self.dynamic_pe_mode == DynamicPeMode.SEXMH:
                positions = self.dynamic_position_encoding_swigluex_multihead(hidden_states, positions)
            else:
                raise NotImplementedError
            q, k = self.rotary_emb.forward_native(positions, q, k, is_dynamic_pe=True)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Olmo2MLP(nn.Module):
    """
    This is the MLP block where the output is computed as
    ``MLP(x)`` in ``LN(MLP(x + LN(Attention(x))))``
    (plus another skip connection).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, Olmo2Config)
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        # Feed-forward input projection.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        # Activation function.
        self.act_fn = SiluAndMul()

        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Olmo2DecoderLayer(nn.Module):
    """
    This is a typical transformer block where the output is
    computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", dynamic_pe_mode: Optional[DynamicPeMode] = DynamicPeMode.N):
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, Olmo2Config)
        # Attention block.
        self.self_attn = Olmo2Attention(vllm_config=vllm_config,
                                        prefix=f"{prefix}.self_attn",
                                        dynamic_pe_mode=dynamic_pe_mode)

        # MLP block.
        self.mlp = Olmo2MLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")

        # LayerNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.post_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                  eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Attention block.
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # MLP block.
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Olmo2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        assert isinstance(self.config, Olmo2Config)

        # pe config
        if hasattr(self.config, "pe_config"):
            pe_config = self.config.pe_config
            dynamic_pe_mode = pe_config.dynamic_pe_mode if hasattr(pe_config, "dynamic_pe_mode") else DynamicPeMode.N
            assert DynamicPeMode.verify(dynamic_pe_mode)
            self.dynamic_pe_mode = dynamic_pe_mode
            self.dynmaic_pe_start_layer = (
                pe_config.dynmaic_pe_start_layer if hasattr(pe_config, "dynmaic_pe_start_layer") else 0
            )
            dynmaic_pe_layer_ids = (
                [int(idx) for idx in pe_config.dynmaic_pe_layer_ids.split(",")]
                if hasattr(pe_config, "dynmaic_pe_layer_ids") and pe_config.dynmaic_pe_layer_ids
                else [idx for idx in range(self.config.num_hidden_layers)]
            )
            dynamic_pe_layer_ids = [idx for idx in dynmaic_pe_layer_ids if idx >= self.dynmaic_pe_start_layer]
        else:
            self.dynamic_pe_mode = DynamicPeMode.N
            dynamic_pe_layer_ids = []

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        dynamic_pe_mode_list = [self.dynamic_pe_mode if i in dynamic_pe_layer_ids else DynamicPeMode.N for i in range(self.config.num_hidden_layers)]
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix, dynamic_pe_mode: Olmo2DecoderLayer(vllm_config=vllm_config,
                                             prefix=prefix, dynamic_pe_mode=dynamic_pe_mode),
            prefix=f"{prefix}.layers",
            dynamic_pe_mode_list=dynamic_pe_mode_list
        )
        print(dynamic_pe_mode_list)
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    self.config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            # Get embeddings of input.
            # shape: (batch_size, seq_len, d_model)
            else:
                hidden_states = self.embed_tokens(input_ids)

        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            assert isinstance(hidden_states, torch.Tensor)

        # Apply blocks one-by-one.
        for layer in self.layers[self.start_layer:self.end_layer]:
            # shape: (batch_size, seq_len, d_model)
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader  # type: ignore
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Olmo2ForCausalLM(nn.Module, SupportsPP):
    """
    Extremely barebones HF model wrapper.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, Olmo2Config)
        self.config = config
        self.model = Olmo2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head.weight"]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
