from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import logging
from ..llama.modeling_llama import LlamaPreTrainedModel, LlamaRMSNorm, eager_attention_forward
from ..olmo.configuration_olmo import OlmoConfig
from ..olmo.modeling_olmo import (
    OlmoAttention,
    OlmoDecoderLayer,
    OlmoForCausalLM,
    OlmoModel,
    OlmoRotaryEmbedding,
    KwargsForCausalLM,
    apply_rotary_pos_emb,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...cache_utils import Cache, DynamicCache, StaticCache
from functools import partial

from ...custom_modules.pe_utils import DynamicPeMode

logger = logging.get_logger(__name__)


class Olmo2Config(OlmoConfig):
    r"""
    This is the configuration class to store the configuration of a [`Olmo2Model`]. It is used to instantiate an OLMo2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/Olmo2-7B-1124-hf](https://huggingface.co/allenai/Olmo2-7B-1124-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the Olmo2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Olmo2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50279):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.

    ```python
    >>> from transformers import Olmo2Model, Olmo2Config

    >>> # Initializing a Olmo2 7B style configuration
    >>> configuration = Olmo2Config()

    >>> # Initializing a model from the Olmo2 7B style configuration
    >>> model = Olmo2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo2"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.rms_norm_eps = rms_norm_eps
        del self.clip_qkv


# OLMo2 RMS norm is identical to Llama RMS norm except:
# - Weight and hidden states are multiplied before converting back to the input dtype, rather than after.
class Olmo2RMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


ALL_LAYERNORM_LAYERS.append(Olmo2RMSNorm)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Olmo2 attention is identical to OLMo attention except:
# - Norm is applied to attention queries and keys.
# - No qkv clipping.
class Olmo2Attention(OlmoAttention):
    def __init__(self, config: Olmo2Config, layer_idx: Optional[int] = None, pe_gen_func: Optional[Callable] = None, dynamic_pe_mode: Optional[DynamicPeMode] = DynamicPeMode.N):
        super().__init__(config, layer_idx=layer_idx)
        self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)
        self.dynamic_pe_mode = dynamic_pe_mode
        if self.dynamic_pe_mode in [DynamicPeMode.C, DynamicPeMode.CWSC]:
            self.dynamic_map = nn.Linear(config.hidden_size, 1, bias=False)
        self.pe_gen_func = pe_gen_func

    def dynamic_position_encoding_continuous(self, hidden_states):
        # hidden_states [B, L, D]
        # pe [B, P, D], where P >= L
        device, dtype = hidden_states.device, hidden_states.dtype
        B, L, D = hidden_states.shape
        pred_indices = self.dynamic_map(hidden_states).squeeze(-1)
        pe = self.pe_gen_func(pred_indices)
        return pred_indices, pe

    def dynamic_position_encoding_attention(self, attn_weights: torch.Tensor, head_id=0):
        # the default head id for dynamic pe is 0
        pred_indices = attn_weights[:, head_id].argmax(dim=-1)
        return pred_indices

    def dynamic_position_encoding_continuous_with_scale_control(self, hidden_states, position_ids):
        # hidden_states [B, L, D]
        # pe [B, P, D], where P >= L
        device, dtype = hidden_states.device, hidden_states.dtype
        B, L, D = hidden_states.shape
        m = F.sigmoid(self.dynamic_map(hidden_states)).squeeze(-1)
        pred_indices = m * position_ids
        pe = self.pe_gen_func(pred_indices)
        return pred_indices, pe

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # get dynamic pe arguments
        dynamic_pe_pdrop = kwargs.get('dynamic_pe_pdrop', None)
        position_ids = kwargs.get('position_ids', None)

        if dynamic_pe_pdrop is not None:
            dynamic_pe_disable = np.random.rand() < dynamic_pe_pdrop
        else:
            dynamic_pe_disable = False

        if not dynamic_pe_disable and self.dynamic_pe_mode == DynamicPeMode.C:
            pred_indices, position_embeddings = self.dynamic_position_encoding_continuous(hidden_states, position_ids)
        elif not dynamic_pe_disable and self.dynamic_pe_mode == DynamicPeMode.CWSC:
            pred_indices, position_embeddings = self.dynamic_position_encoding_continuous_with_scale_control(hidden_states, position_ids)
        elif not dynamic_pe_disable and self.dynamic_pe_mode == DynamicPeMode.A:
            position_embeddings = self.pe_gen_func(position_ids)
        else:
            pred_indices = torch.arange(hidden_states.shape[1]).unsqueeze(0)
            pred_indices = pred_indices.repeat(hidden_states.shape[0], 1)

        cos, sin = position_embeddings[0].to(hidden_states.dtype), position_embeddings[1].to(hidden_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False) and self.dynamic_pe_mode != DynamicPeMode.A:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.dynamic_pe_mode == DynamicPeMode.A:
            # attention based dynamic pe get i+1 layer's pred_indices from i-th layer
            pred_indices = self.dynamic_position_encoding_attention(attn_weights)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, pred_indices


# The OLMo2 layers are identical to those of the OLMo model except:
# - RMSNorm is used instead of standard layer norm.
# - Norm is applied after attention/feedforward rather than before.
class Olmo2DecoderLayer(OlmoDecoderLayer):
    def __init__(self, config: Olmo2Config, layer_idx: int, pe_gen_func: Optional[Callable] = None, dynamic_pe_mode: Optional[DynamicPeMode] = DynamicPeMode.N):
        super().__init__(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx, pe_gen_func=pe_gen_func, dynamic_pe_mode=dynamic_pe_mode)
        del self.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, pred_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if kwargs.get('output_pred_indices', False):
            outputs += (pred_indices,)

        return outputs


class Olmo2RotaryEmbedding(OlmoRotaryEmbedding):
    def __init__(self, config: Olmo2Config, device=None):
        super().__init__(config, device)
    
    def from_positions(self, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = position_ids.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(position_ids.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos, sin


class Olmo2PreTrainedModel(LlamaPreTrainedModel):
    pass


# The OLMo2 model is identical to the OLMo model, except RMSNorm is used instead of
# standard layer norm for the output norm.
class Olmo2Model(OlmoModel):
    def __init__(self, config: Olmo2Config):
        super().__init__(config)
        self.norm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Olmo2RotaryEmbedding(config=config)
        # pe config
        if hasattr(self.config, 'pe_config'):
            pe_config = self.config.pe_config
            dynamic_pe_mode = pe_config.dynamic_pe_mode if hasattr(pe_config, 'dynamic_pe_mode') else DynamicPeMode.N
            assert DynamicPeMode.verify(dynamic_pe_mode)
            self.dynamic_pe_mode = dynamic_pe_mode
            self.dynmaic_pe_start_layer = pe_config.dynmaic_pe_start_layer if hasattr(pe_config, 'dynmaic_pe_start_layer') else 0
            dynmaic_pe_layer_ids = [int(idx) for idx in pe_config.dynmaic_pe_layer_ids.split(",")]  if hasattr(pe_config, 'dynmaic_pe_layer_ids') and pe_config.dynmaic_pe_layer_ids else [idx for idx in range(config.num_hidden_layers)]
            dynamic_pe_layer_ids = [idx for idx in dynmaic_pe_layer_ids if idx >= self.dynmaic_pe_start_layer]
        else:
            self.dynamic_pe_mode = DynamicPeMode.N
            dynamic_pe_layer_ids = []

        self.layers = nn.ModuleList(
            [Olmo2DecoderLayer(
                config, layer_idx, self.rotary_emb.from_positions,
                dynamic_pe_mode=self.dynamic_pe_mode if layer_idx in dynamic_pe_layer_ids else DynamicPeMode.N) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_pred_indices:  Optional[bool] = None,
        dynamic_pe_pdrop: Optional[float] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_pred_indices = () if output_pred_indices else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    output_pred_indices,
                    dynamic_pe_pdrop
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_pred_indices=output_pred_indices,
                    dynamic_pe_pdrop=dynamic_pe_pdrop
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_pred_indices:
                all_pred_indices += (layer_outputs[-1],)
            if i >= self.dynmaic_pe_start_layer and self.dynamic_pe_mode in [DynamicPeMode.A]:
                position_ids = layer_outputs[-1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            pred_indices=all_pred_indices if all_pred_indices else None,
        )


# The heads now only need to redefine the model inside to the correct `RobertaModel`
class Olmo2ForCausalLM(OlmoForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Olmo2ForCausalLM

        >>> model = Olmo2ForCausalLM.from_pretrained("meta-olmo2/Olmo2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-olmo2/Olmo2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pred_indices=outputs.pred_indices
        )


__all__ = [
    "Olmo2Config",
    "Olmo2ForCausalLM",
    "Olmo2Model",
    "Olmo2PreTrainedModel",  # noqa: F822
]
