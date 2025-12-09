import math
import torch
from torch import nn
from torch.nn.init import trunc_normal_, normal_
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F
from .pytorch_utils import Conv1D
from .activations import ACT2FN
from torch.cuda.amp import autocast
import numpy as np
from .pe_utils import PeUtils

INIT_NORM_WEIGHT = 1.0


class TableEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        self.end_x, self.end_y = int(math.sqrt(num_positions)), int(math.sqrt(num_positions))
        super().__init__(num_positions, embedding_dim)
        self.init_weights()

    def init_weights(self):
        normal_(self.weight, mean=0, std=0.02)
    
    @classmethod
    def from_pretrained(cls, pretrained_path, freeze_pretrained_pe=True):
        weight = torch.load(pretrained_path, map_location='cpu')
        num_positions, embedding_dim = weight.shape
        table_pe = cls(num_positions, embedding_dim)
        table_pe.weight.data.copy_(weight)
        if freeze_pretrained_pe:
            table_pe.weight.requires_grad = False
            table_pe.weight.detach_()
        return table_pe

    def forward(self, pos_ids) -> torch.Tensor:
        if pos_ids.dim() > 1:
            pos_ids = pos_ids[:, 0] * self.end_x + pos_ids[:, 1]
        return super().forward(pos_ids)


class AlibiPositionalEmbedding(nn.Embedding):
    """
    Adapted from the offical implementation of ALIBI: https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py
    """
    def __init__(self, num_positions: int, num_heads: int=12, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, num_heads)
        self.skip_random_init = True
        self.weight = self._init_weight(self.weight)
        self.skip_random_init = True
    
    @staticmethod
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
            return get_slopes_power_of_2(closest_power_of_2) + AlibiPositionalEmbedding.get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        num_positions, num_heads = out.shape
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        slopes = torch.FloatTensor(AlibiPositionalEmbedding.get_slopes(num_heads))
        alibi = slopes.unsqueeze(1) * torch.arange(num_positions).unsqueeze(0).expand(num_heads, -1)
        out[:, :] = alibi.transpose(0, 1)
        out.detach_()
        return out

    @staticmethod
    def from_positions(positions: torch.LongTensor, num_heads: int) -> torch.Tensor:
        # positions: [B, T]
        slopes = torch.tensor(
            AlibiPositionalEmbedding.get_slopes(num_heads),
            dtype=torch.float32,
            device=positions.device
        )
        pe = slopes.unsqueeze(1).unsqueeze(1) * positions.unsqueeze(0).expand(num_heads, -1, -1)
        return pe.permute(1, 2, 0)

    @torch.no_grad()
    def forward(self, pos_ids) -> torch.Tensor:
        if pos_ids.dim() > 1:
            pos_ids = pos_ids[:, 0] * self.end_x + pos_ids[:, 1]
        alibi = super().forward(pos_ids)
        return alibi

class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, base=10000, padding_idx: Optional[int] = None, init_scale=1.0) -> None:
        self.end_x, self.end_y = int(math.sqrt(num_positions)), int(math.sqrt(num_positions))
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight, base=base, init_scale=init_scale)
        self.base = base
        self.skip_random_init = True

    def get_scaled_base(self, base, scale, dim):
        return base * scale ** (dim / (dim - 2))

    @staticmethod
    def from_positions(positions: torch.LongTensor, dim: float = 768, base: float = 10000, init_scale=1.0) -> torch.Tensor:
        B, T = positions.shape
        device = positions.device

        freq_seq = torch.arange(dim // 2, device=device)
        inv_freq = base ** (-2 * freq_seq / dim)
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0) # [B, T, dim // 2]
        pe = torch.zeros(B, T, dim, device=device)
        pe[..., 0::2] = init_scale * torch.sin(angles)
        pe[..., 1::2] = init_scale * torch.cos(angles)
        return pe

    @staticmethod
    def _init_weight(out: nn.Parameter, base = 4300, init_scale=1.0) -> nn.Parameter:
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(base, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        out[:, 0::2] = torch.FloatTensor(init_scale * np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(init_scale * np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, pos_ids) -> torch.Tensor:
        if pos_ids.dim() > 1:
            pos_ids = pos_ids[:, 0] * self.end_x + pos_ids[:, 1]
        return super().forward(pos_ids)


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, base=10000, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight, base=base)
        self.base = base
        self.skip_random_init = True

    def get_scaled_base(self, base, scale, dim):
        return base * scale ** (dim / (dim - 2))

    @staticmethod
    def from_positions(positions: torch.LongTensor, dim: float = 64, base: float = 10000) -> torch.Tensor:
        assert dim % 2 == 0, "hidden size `dim` must be even"
        freq_seq = torch.arange(dim // 2, dtype=positions.dtype, device=positions.device)
        inv_freq = base ** (-2 * freq_seq / dim)             # (half,)
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        sin_part = torch.sin(angles)                       # [B, T, half]
        cos_part = torch.cos(angles)                       # [B, T, half]
        pe = torch.cat([sin_part, cos_part], dim=-1)       # [B, T, D]
        return pe

    @staticmethod
    def _init_weight(out: nn.Parameter, base = 4300) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(base, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, pos_ids) -> torch.Tensor:
        return super().forward(pos_ids)


class RoFormer2DMixedPositionalEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int, base=100, vit_depth=12, num_heads=6, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.end_x, self.end_y = int(math.sqrt(num_positions)), int(math.sqrt(num_positions))
        freqs = []
        self.vit_depth = vit_depth
        self.num_heads = num_heads
        for i in range(vit_depth):
            freqs.append(
                self.init_random_2d_freqs(dim=embedding_dim, num_heads=num_heads, theta=base)
            )
        freqs = torch.stack(freqs, dim=1).view(2, vit_depth, -1)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
        self.freqs_dtype = self.freqs.dtype
        self.skip_random_init = True

    def init_random_2d_freqs(self, dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
        freqs_x = []
        freqs_y = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # [head_dim // 4]
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1) # [head_dim // 2]
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1) # [head_dim // 2]
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0) # [num_heads, head_dim // 2]
        freqs_y = torch.stack(freqs_y, dim=0) # [num_heads, head_dim // 2]
        freqs = torch.stack([freqs_x, freqs_y], dim=0) # [2, num_heads, head_dim // 2]
        return freqs

    def get_scaled_base(self, base, scale, dim):
        return base * scale ** (dim / (dim - 2))
    
    def compute_mixed_cis(self, t_x, t_y):
        N = t_x.shape[0]
        depth = self.freqs.shape[1]
        
        # No float 16 for this range
        with torch.cuda.amp.autocast(enabled=False):
            freqs_x = (t_x.unsqueeze(-1) @ self.freqs[0].unsqueeze(-2)).view(depth, N, self.num_heads, -1).permute(0, 2, 1, 3)
            freqs_y = (t_y.unsqueeze(-1) @ self.freqs[1].unsqueeze(-2)).view(depth, N, self.num_heads, -1).permute(0, 2, 1, 3)
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
        return freqs_cis

    def forward(self, pos_ids) -> torch.Tensor:
        # pos_ids is in shape [h, w] -> [t_y, t_x]
        pos_ids = pos_ids.to(self.freqs_dtype)
        return self.compute_mixed_cis(pos_ids[:, 1], pos_ids[:, 0])


class RoFormer2DPositionalEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int, base=100, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.end_x, self.end_y = int(math.sqrt(num_positions)), int(math.sqrt(num_positions))
        weight = self._init_weight(num_positions, embedding_dim, base=base)
        self.register_buffer('weight', weight)
        self.skip_random_init = True

    def get_scaled_base(self, base, scale, dim):
        return base * scale ** (dim / (dim - 2))

    def _init_weight(self, n_pos, dim, base = 100):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        t = torch.arange(n_pos, dtype=torch.float32)
        t_x = (t % self.end_x).float()
        t_y = torch.div(t, self.end_x, rounding_mode='floor').float()   
     
        freqs_x = 1.0 / (base ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        freqs_y = 1.0 / (base ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

        freqs_x = torch.outer(t_x, freqs_x) # [N_x, head_dim // 4]
        freqs_y = torch.outer(t_y, freqs_y) # [N_y, head_dim // 4]
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x) # [N_x, head_dim // 4] dtype=complex 
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y) # [N_y, head_dim // 4] dtype=complex 
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    @torch.no_grad()
    def forward(self, pos_ids) -> torch.Tensor:
        if pos_ids.dim() > 1:
            pos_ids = pos_ids[:, 0] * self.end_x + pos_ids[:, 1]
        return self.weight[pos_ids]


def _init_seqpe_vit_weights(module: nn.Module):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, Conv1D) or isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        trunc_normal_(module.weight, std=.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.constant(module.weight, INIT_NORM_WEIGHT)

def _init_seqpe_gpt_weights(module, initializer_range=0.02):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, Conv1D)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(INIT_NORM_WEIGHT)


class SequentialPeAttention(nn.Module):
    def __init__(self, num_attention_heads, embed_dim, attn_pdrop=0.1, resid_pdrop=0.1, scale_attn_weights=True):
        super().__init__()
        self.num_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = scale_attn_weights
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], (value.size(-1)) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            mask_value = torch.finfo(attn_weights.dtype).min
            attention_mask = mask_value * (1-attention_mask)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]: 
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
class SequentialPeMLP(nn.Module):
    def __init__(self, intermediate_size, hidden_size, activation_function='gelu_new', resid_pdrop=0.1):
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, intermediate_size)
        self.act = ACT2FN[activation_function]
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states) ### !!!!!!!
        return hidden_states
    

class SequentialPeBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, activation_function='gelu_new',
                 resid_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05,
                 scale_attn_weights=True):
        super().__init__()
        inner_dim = 4 * hidden_size
        self.attn = SequentialPeAttention(
            num_attention_heads=num_attention_heads, embed_dim=hidden_size,
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,scale_attn_weights=scale_attn_weights
        )
        self.layernorm_1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.layernorm_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        # self.layernorm_final = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = SequentialPeMLP(
            inner_dim, hidden_size=hidden_size,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop
        )
        
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        seq_pe = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]: ### ???
        
        residual = hidden_states
        hidden_states = self.layernorm_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states

        hidden_states = self.layernorm_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        # hidden_states = self.layernorm_final(residual + feed_forward_hidden_states) ### !!!
        hidden_states = residual + feed_forward_hidden_states
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs 

class SequentialPE(nn.Module):
    def __init__(self, pe_embed_dim, num_attention_heads, layer_num, max_digits,
                 attn_direction='causal', use_last_layernorm=False, mask_padding=True,
                 activation_function='gelu_new', resid_pdrop=0.1, attn_pdrop=0.1,
                 layer_norm_epsilon=1e-05, scale_attn_weights=True, use_cls_token=True, 
                 out_proj_dim=-1, seqpe_temperature=1.0, seqpe_init_norm_weight=1.0, data_dim=1):
        super(SequentialPE, self).__init__()
        self.max_digits = max_digits
        self.pe_embed_dim = pe_embed_dim
        self.head_num = num_attention_heads
        self.layer_num = layer_num
        self.mask_padding = mask_padding
        self.data_dim = data_dim
        self.attn_direction = attn_direction
        self.use_cls_token = use_cls_token
        self.seqpe_temperature = seqpe_temperature
        self.use_last_layernorm = use_last_layernorm
        global INIT_NORM_WEIGHT
        INIT_NORM_WEIGHT = seqpe_init_norm_weight
        # self.seqpe_norm_scale = seqpe_norm_scale
        if self.attn_direction == 'bi':
            # NOTE: both causal and bi attn can use cls token, and it is mandatory for bi attn.
            assert self.use_cls_token

        self.pos_of_seqpe_embed = nn.Embedding(self.max_digits * self.data_dim, self.pe_embed_dim)
        # TODO: It doesn't necessarily have to be decimal; it can also be hexadecimal, base-26, and so on.
        self.token_of_seqpe_embed = nn.Embedding(10, self.pe_embed_dim)
        if data_dim > 1:
            self.dim_token_embed = nn.Parameter(torch.zeros(data_dim, 1, self.pe_embed_dim))
        
        if self.use_cls_token:
            self.cls_token_embed = nn.Parameter(torch.zeros(1, 1, self.pe_embed_dim))

        self.transformers = nn.ModuleList(
            [
                SequentialPeBlock(
                    hidden_size=pe_embed_dim, num_attention_heads=self.head_num, activation_function=activation_function,
                    resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop, layer_norm_epsilon=layer_norm_epsilon,
                    scale_attn_weights=scale_attn_weights,
                )
                for i in range(self.layer_num)
            ]
        )
        if self.use_last_layernorm:
            self.layernorm = nn.LayerNorm(pe_embed_dim, eps=layer_norm_epsilon)
        self.out_proj = nn.Linear(self.pe_embed_dim, out_proj_dim if out_proj_dim > 0 else self.pe_embed_dim)
        self.init_weights()

    def init_weights(self):
        if self.data_dim > 1:
            normal_(self.dim_token_embed, std=.02)
        if self.use_cls_token:
            normal_(self.cls_token_embed, std=.02)
        self.apply(_init_seqpe_gpt_weights)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        name_list = ['pos_of_seqpe_embed', 'token_of_seqpe_embed', 'layernorm']
        if self.data_dim > 1:
            name_list += ['dim_token_embed']
        if self.use_cls_token:
            name_list += ['cls_token_embed']
        return set(name_list)
    
    def _get_dim_embed(self, batch_size, n_digits):
        n_dim = self.dim_token_embed.size(0)
        n_digits = n_digits // n_dim
        x = self.dim_token_embed.expand(-1, n_digits, -1).flatten(0, 1)
        x = x.unsqueeze(0).expand(batch_size, -1, -1)
        return x

    def forward(self, x, pad_mask):
        pos_of_x = PeUtils.get_pos_of_seqpe(x)
        attention_mask = PeUtils.get_seqpe_mask(
            x, pad_mask, attn_mode=self.attn_direction,
            mask_padding=self.mask_padding, add_cls_mask=self.use_cls_token
        )
        x = self.token_of_seqpe_embed(x)
        pos_of_x = self.pos_of_seqpe_embed(pos_of_x)
        
        x = x + pos_of_x

        if self.data_dim > 1:
            batch_size, n_digits = x.shape[:2]
            x = x + self._get_dim_embed(batch_size, n_digits)

        if self.use_cls_token:
            # NOTE: cls_token is at last and without PE
            x = torch.cat([x, self.cls_token_embed.expand(x.shape[0], -1, -1)], dim=1)

        for i, pe_transformer_block in enumerate(self.transformers):
            x = pe_transformer_block(x, attention_mask=attention_mask)[0]
        if self.use_last_layernorm:
            x = self.layernorm(x)
        x = self.out_proj(x[:, -1]) * (1 / self.seqpe_temperature)
        return x