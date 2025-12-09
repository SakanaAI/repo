import pdb
import torch
import math
import numpy as np
import torch.nn.functional as F
import json


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
    SEXSIG = 'swigluex_sigmoid'
    SEXMH = 'swigluex_multihead'
    SEXPA = 'swigluex_posavg'
    SEXFEX = 'swigluex_filterex'

    @staticmethod
    def verify(x):
        return x in ['none', 'multihead_continuous', 'continuous', 'discrete', 'continuous_with_scale_control', 'attention', 'swiglu', 'swigluex', 'nope', 'swiglu_scientific', 'swiglu_scientific_tanh', 'swigluex_posdim', 'swigluex_filterex', 'swigluex_posavg', 'swigluex_multihead', 'swigluex_sigmoid']


class PeUtils:
    @staticmethod
    def dump_jsonl(data, fname):
        with open(fname, 'w') as fout:
            for it in data:
                fout.write(json.dumps(it) + "\n")

    @staticmethod
    def get_dynamic_pe_dropout(init_pdrop, cur_step, max_step):
        if max_step == 0 or cur_step > max_step:
            return 0.0
        else:
            delta_pdrop = init_pdrop / max_step
            return init_pdrop - delta_pdrop * cur_step    

    @staticmethod
    def get_dynamic_pe_anneal_temp(init_temp, end_temp, cur_step, max_step):
        assert init_temp > end_temp
        if cur_step > max_step:
            return end_temp
        else:
            step_temp = (init_temp - end_temp) / max_step
            return init_temp - step_temp * cur_step

    @staticmethod
    def adapt_for_attn_mask(attn_mask, pe_attn_scalar, scale):
        # attn_mask: [B, H, N_q, N_k]
        # pe_attn_scalar: [B, H, N_q, N_k]
        # This function is to adapt the pe_attn_scalar information into the attn_mask
        # This function allows the use of memory-efficient attention functions, e.g., sdpa
        # The general attn pipeline: 1) dot product -> 2) scale -> 3) attn_mask -> 4) value mapping
        # The pe_attn_scalar usualy happens between (1) and (2), but now we want to move it to (3)

        pe_attn_scalar = pe_attn_scalar / scale # [B, H, N]
        attn_mask = attn_mask +  pe_attn_scalar
        return attn_mask

    @staticmethod
    def get_digit_num(max_sample_position, base=10):
        s = str(max_sample_position-1)
        return len(s)

    @staticmethod
    def merge_pe_data(main_sample, ct_sample=None, trans_sample=None):
        main_pos_seq = main_sample['pos_seq_data']
        main_pad_mask = main_sample['pad_mask']
        sizes = [main_pos_seq.size(0)]
        pos_seq = [main_pos_seq]
        pad_mask = [main_pad_mask]
        if ct_sample is not None:
            pivot_pos_seq = ct_sample['pivot_seq_data']
            pivot_pad_mask = ct_sample['pivot_pad_mask']
            neg_seq_data = ct_sample['neg_seq_data']
            neg_pad_mask = ct_sample['neg_pad_mask']
            sizes += [pivot_pos_seq.size(0), neg_seq_data.size(0)]
            pos_seq += [pivot_pos_seq, neg_seq_data]
            pad_mask += [pivot_pad_mask, neg_pad_mask]
        if trans_sample is not None:
            trans_seq_data = trans_sample['pos_seq_data']
            trans_pad_mask = trans_sample['pad_mask']
            sizes += [trans_seq_data.size(0)]
            pos_seq += [trans_seq_data]
            pad_mask += [trans_pad_mask]

        pos_seq = torch.cat(pos_seq, dim=0)
        pad_mask = torch.cat(pad_mask, dim=0)
        return pos_seq, pad_mask, sizes

    @staticmethod
    def split_pe_data(pe, sizes):
        split_pe = torch.split(pe, sizes, dim=0)
        return split_pe

    @staticmethod
    def pos2seq(number, max_digits, pad_value, is_left_padding=True):
        """
        Convert an integer position to a sequence.
        We use the left padding, i.e., 123 -> [0,0,1,2,3]
        """
        # TODO: It doesn't necessarily have to be decimal; it can also be hexadecimal, base-26, and so on.
        num_str = str(number)
        num_list = [int(d) for d in num_str]
        num_zeros = max_digits - len(num_list)
        if is_left_padding:
            pad_mask = [0] * num_zeros + [1] * len(num_list)
            return [pad_value] * num_zeros + num_list, pad_mask
        else:
            pad_mask = [1] * len(num_list) + [0] * num_zeros
            return num_list + [pad_value] * num_zeros, pad_mask
    
    @staticmethod
    def prepare_linear_interpolation(orig_left, orig_right, new_left, new_right, x):
        if isinstance(orig_left, int):
            orig_range = orig_right - orig_left
            new_range = new_right - new_left
            new_x = orig_range / new_range * x
            left_x, right_x = math.floor(new_x), math.ceil(new_x)
            d = new_x - left_x
            return (left_x, right_x), (d, 1-d)
        elif isinstance(orig_left, tuple):
            orig_range = (orig_right[0] - orig_left[0], orig_right[1] - orig_left[1])
            new_range = (new_right[0] - new_left[0], new_right[1] - new_left[1])
            new_x = (orig_range[0] / new_range[0] * x[0], orig_range[1] / new_range[1] * x[1])
            left_x, right_x = math.floor(new_x[0]), math.ceil(new_x[0])
            left_y, right_y = math.floor(new_x[1]), math.ceil(new_x[1])
            dx = new_x[0] - left_x
            dy = new_x[1] - left_y
            return ((left_x, left_y), (left_x, right_y), (right_x, left_y), (right_y, right_y)), (dx*dy, dx*(1-dy), (1-dx)*dy, (1-dx)*(1-dy))
        else:
            raise NotImplementedError

    @staticmethod
    def full_interpolation(pos_embed, orig_shape, new_shape, mode):
        if mode in ['bicubic', 'bilinear']:
            assert isinstance(orig_shape, tuple) and len(orig_shape) == len(new_shape)
            pos_embed = pos_embed.view(
                1, orig_shape[0], orig_shape[1], pos_embed.size(-1)
            ).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=new_shape, mode=mode, align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        elif mode == "linear":
            assert isinstance(orig_shape, int) and isinstance(new_shape, int) 
            pos_embed = pos_embed.view(
                1, orig_shape, pos_embed.size(-1)
            ).permute(0, 2, 1)
            pos_embed = F.interpolate(
                pos_embed, size=new_shape, mode='linear', align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 1).flatten(0, 1)
        else:
            raise NotImplementedError
        return pos_embed
    
    # @staticmethod
    # def create_2d_pos_list(start, end, data_dim=2):
    #     # a mesh from (start, start) to (end, end)
    #     range_list = [range(start, end) for _ in range(data_dim)]
    #     wv, hv = np.meshgrid(range_list[0], range_list[1])
    #     # data like [(0, 0), ... (0, W), (1, 0) ... (1, W), ...]
    #     pos_list = np.column_stack([hv.ravel(), wv.ravel()]).tolist()
    #     return [tuple(it) for it in pos_list]

    @staticmethod
    def create_2d_pos_list(h1, w1, h2, w2):
        assert h1 < h2 and w1 < w2
        h_range_list = range(h1, h2)
        w_range_list = range(w1, w2)
        wv, hv = np.meshgrid(w_range_list, h_range_list)
        # data like [(h1, w1), ... (h1, w2), (h1+1, w1) ... (1, W), ...]
        pos_list = np.column_stack([hv.ravel(), wv.ravel()]).tolist()
        return [tuple(it) for it in pos_list]
    
    @staticmethod
    def prepare_seqpe_data(raw_pos_list, max_digits, device=torch.device("cpu"), pad_value=0, is_left_padding=True):
        pad_mask_list, padded_pos_list = [], []
        for pos in raw_pos_list:
            if isinstance(pos, int):
                padded_pos, pad_mask = PeUtils.pos2seq(pos, max_digits, pad_value, is_left_padding)
            else:
                padded_pos, pad_mask = [], []
                for sub_pos in pos:
                    sub_padded_pos, sub_pad_mask = PeUtils.pos2seq(sub_pos, max_digits, pad_value, is_left_padding)
                    padded_pos += sub_padded_pos
                    pad_mask += sub_pad_mask
            pad_mask_list.append(pad_mask)
            padded_pos_list.append(padded_pos)
        padded_tensor = torch.tensor(padded_pos_list, dtype=torch.long, device=device)
        pad_mask_tensor = torch.tensor(pad_mask_list, dtype=torch.float, device=device)
        return padded_tensor, pad_mask_tensor
    
    @staticmethod
    def distance_1d_data(x, y):
        return abs(x-y)

    @staticmethod
    def distance_2d_data(x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    @staticmethod
    def get_pos_of_seqpe(x):
        """
        When modelling the PE as a sequence, we also need to represent the pos of each digit in sequence
        """
        n_batch, n_max_len = x.size(0), x.size(1)
        device = x.device
        pos_of_seqpe = torch.arange(n_max_len, device=device, dtype=int).unsqueeze(0)
        pos_of_seqpe = pos_of_seqpe.repeat(n_batch, 1)
        return pos_of_seqpe
    
    @staticmethod
    def get_seqpe_mask(x, pad_mask, attn_mode='causal', mask_padding=True, add_cls_mask=False):
        n_batch, n_max_len = x.size(0), x.size(1)
        device = x.device
        assert isinstance(add_cls_mask, bool)
        add_cls_mask = int(add_cls_mask)
        if attn_mode == 'bi':
            attn_mask = torch.ones((n_max_len+add_cls_mask, n_max_len+add_cls_mask), device=device).unsqueeze(0)
            attn_mask = attn_mask.repeat(n_batch, 1, 1)
        else:
            attn_mask = torch.tril(torch.ones((n_max_len+add_cls_mask, n_max_len+add_cls_mask), device=device)).unsqueeze(0)
            attn_mask = attn_mask.repeat(n_batch, 1, 1)
        mask = attn_mask
        if mask_padding:
            pad_mask = torch.cat([pad_mask, torch.ones(pad_mask.size(0), add_cls_mask, dtype=pad_mask.dtype, device=device)], dim=-1)
            mask = attn_mask * pad_mask.unsqueeze(1) # logical and

        return mask.float()
    
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
               
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

    @staticmethod
    def apply_rotary2d_position_embeddings(freqs_cis: torch.Tensor, xq: torch.Tensor, xk: torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
    

def _test_create_2d_pos_list():
    p1, p2 = (0, 0), (5, 5)
    print(f"_test_create_2d_pos_list | ({p1[0]}, {p1[1]}) and ({p2[0]}, {p2[1]})")
    ans = PeUtils.create_2d_pos_list(p1[0], p1[1], p2[0], p2[1])
    print(ans)
    print(len(ans))

    p1, p2 = (7, 8), (12, 13)
    print(f"_test_create_2d_pos_list | ({p1[0]}, {p1[1]}) and ({p2[0]}, {p2[1]})")
    ans = PeUtils.create_2d_pos_list(p1[0], p1[1], p2[0], p2[1])
    print(ans)
    print(len(ans))

    p1, p2 = (0, 0), (8, 4)
    print(f"_test_create_2d_pos_list | ({p1[0]}, {p1[1]}) and ({p2[0]}, {p2[1]})")
    ans = PeUtils.create_2d_pos_list(p1[0], p1[1], p2[0], p2[1])
    print(ans)
    print(len(ans))


def _test_prepare_seqpe_data_and_get_pos_of_seqpe(is_left_padding):
    max_digits=5
    data_dim=1
    raw_pos_list = [1, 12, 123, 1234, 12345]
    print(f"_test_prepare_seqpe_data | max_digits={max_digits} | is_left_padding={is_left_padding} | data_dim={data_dim}")
    data = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits, is_left_padding=is_left_padding)
    print(data[0])
    print(data[1])
    pos_of_seqpe = PeUtils.get_pos_of_seqpe(data[0])
    print(pos_of_seqpe)
    print("-" * 50)

    data_dim=2
    raw_pos_list = [(1, 1), (12, 12), (123, 1), (1234, 1234), (12345, 123)]
    print(f"_test_prepare_seqpe_data | max_digits={max_digits} | is_left_padding={is_left_padding} | data_dim={data_dim}")
    data = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits, is_left_padding=is_left_padding)
    print(data[0])
    print(data[1])
    pos_of_seqpe = PeUtils.get_pos_of_seqpe(data[0])
    print(pos_of_seqpe)
    print("-" * 50)


def _test_get_seqpe_mask_add_cls(attn_mode, mask_padding):
    max_digits=5
    data_dim=1
    raw_pos_list = [1, 12, 123, 1234, 12345]
    print(f"_test_get_seqpe_mask | attn_mode={attn_mode} | mask_padding={mask_padding} | data_dim={data_dim} | add_cls_mask=True")
    data = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits)
    mask = PeUtils.get_seqpe_mask(data[0], data[1], attn_mode=attn_mode, mask_padding=mask_padding, add_cls_mask=True)
    print(data[0])
    print(mask)
    print("-" * 50)

    data_dim=2
    raw_pos_list = [(1, 1), (12, 12), (123, 1), (1234, 1234), (12345, 123)]
    print(f"_test_get_seqpe_mask | attn_mode={attn_mode} | mask_padding={mask_padding} | data_dim={data_dim} | add_cls_mask=True")
    data = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits)
    mask = PeUtils.get_seqpe_mask(data[0], data[1], attn_mode=attn_mode, mask_padding=mask_padding, add_cls_mask=True)
    print(data[0])
    print(mask)
    print("-" * 50)

def _test_get_seqpe_mask(attn_mode, mask_padding):
    max_digits=5
    data_dim=1
    raw_pos_list = [1, 12, 123, 1234, 12345]
    print(f"_test_get_seqpe_mask | attn_mode={attn_mode} | mask_padding={mask_padding} | data_dim={data_dim}")
    data = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits)
    mask = PeUtils.get_seqpe_mask(data[0], data[1], attn_mode=attn_mode, mask_padding=mask_padding)
    print(data[0])
    print(mask)
    print("-" * 50)

    data_dim=2
    raw_pos_list = [(1, 1), (12, 12), (123, 1), (1234, 1234), (12345, 123)]
    print(f"_test_get_seqpe_mask | attn_mode={attn_mode} | mask_padding={mask_padding} | data_dim={data_dim}")
    data = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits)
    mask = PeUtils.get_seqpe_mask(data[0], data[1], attn_mode=attn_mode, mask_padding=mask_padding)
    print(data[0])
    print(mask)
    print("-" * 50)


if __name__ == "__main__":
    _test_create_2d_pos_list()
    # _test_prepare_seqpe_data_and_get_pos_of_seqpe(True)
    # _test_prepare_seqpe_data_and_get_pos_of_seqpe(False)
    # _test_get_seqpe_mask('causal', True)
    # _test_get_seqpe_mask('causal', False)
    # _test_get_seqpe_mask('bi', True)
    # _test_get_seqpe_mask('bi', False)

    _test_get_seqpe_mask_add_cls('causal', True)
    _test_get_seqpe_mask_add_cls('causal', False)
    _test_get_seqpe_mask_add_cls('bi', True)
    _test_get_seqpe_mask_add_cls('bi', False)