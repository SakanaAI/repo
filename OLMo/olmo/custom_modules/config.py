from yacs.config import CfgNode as CN
import os
from itertools import chain

def save_pe_config(config, save_dir):
    path = os.path.join(save_dir, "pe_config.json")
    with open(path, "w") as f:
        f.write(config.dump())


def get_pe_config(args):
    # pe config
    config = CN()
    config.defrost()
    config.MODEL = CN()
    config.PE_EMBED_DIM = args.pe_embed_dim
    config.NUM_ATTENTION_HEADS = args.num_attention_heads
    config.PE_TYPE = args.pe_type
    config.PE_APPLY_METHOD = args.pe_apply_method
    config.PE_DATA_DIM = args.pe_data_dim
    config.PE_MAX_POSITION = args.pe_max_position
    config.PE_USE_RANDOM_SHIFT = args.pe_use_random_shift
    config.PE_RANDOM_SHIFT_RATE = args.pe_random_shift_rate
    config.PE_RANDOM_SHIFT_DOWNSAMPLE = args.pe_random_shift_downsample
    config.SINUSOIDAL_PE_BASE = args.sinusoidal_pe_base
    config.SINUSOIDAL_PE_SCALE = args.sinusoidal_pe_scale
    config.USE_PE_MULTI_HEAD = args.use_pe_multi_head
    # config.SEQPE_MULTI_HEAD_LOSS = args.seqpe_multi_head_loss
    config.USE_PE_QK_PER_LAYER = args.use_pe_qk_per_layer
    config.SEQPE_SCALAR_SCALE = args.seqpe_scalar_scale
    config.DYNAMIC_PE_MODE = args.dynamic_pe_mode
    config.USE_GUMBLE = args.use_gumble
    config.DYNAMIC_PE_TEMP = args.dynamic_pe_temp
    config.DYNAMIC_PE_INIT_PDROP = args.dynamic_pe_init_pdrop
    config.DYNAMIC_PE_DROPOUT_ANNEAL_STEP = args.dynamic_pe_dropout_anneal_step
    config.USE_STE = args.use_ste
    config.TABLEPE_PRETRAINED = args.tablepe_pretrained
    config.FREEZE_TABLEPE = args.freeze_tablepe
    config.DYNMAIC_PE_START_LAYER = args.dynmaic_pe_start_layer
    config.DYNMAIC_PE_LAYER_IDS = args.dynmaic_pe_layer_ids
    config.DYNAMIC_PE_MOVING_RATIO = args.dynamic_pe_moving_ratio
    config.DYNAMIC_PE_MOVING_RATIO = args.dynamic_pe_moving_ratio
    config.DYNAMIC_PE_MOVING_BLOCKS = args.dynamic_pe_moving_blocks

    if hasattr(args, 'per_device_train_batch_size'):
        batch_size = args.per_device_train_batch_size
    elif hasattr(args, 'per_device_eval_batch_size'):
        batch_size = args.per_device_train_batch_size
    else:
        raise ValueError

    config.PE_MAIN_BATCH_SIZE = batch_size if args.pe_main_batch_size < 0 else min(args.pe_main_batch_size, batch_size)
    if args.pe_use_random_shift:
        assert (batch_size % config.PE_MAIN_BATCH_SIZE) == 0
    
    if config.PE_APPLY_METHOD == "input_add":
        out_proj_dim = config.PE_EMBED_DIM
    elif config.PE_APPLY_METHOD in ["rotary", "rotary_2d", "rotary_mixed"]:
        out_proj_dim = config.PE_EMBED_DIM // config.NUM_ATTENTION_HEADS
    elif config.PE_APPLY_METHOD.startswith("attn"):
        if config.USE_PE_MULTI_HEAD:
            # proj_dim = head_dim * head
            out_proj_dim = config.PE_EMBED_DIM
        else:
            # proj_dim = head_dim
            out_proj_dim = config.PE_EMBED_DIM // config.NUM_ATTENTION_HEADS
    elif config.PE_TYPE == "alibi" or config.PE_APPLY_METHOD == "nope":
        out_proj_dim = -1
    else:
        raise NotImplementedError
    config.PE_OUT_PROJ_DIM = out_proj_dim

    # seqpe config
    config.SEQPE_DIST_SAMPLE_RANGE = args.seqpe_dist_sample_range
    config.SEQPE_MAX_DIGITS = args.seqpe_max_digits
    config.SEQPE_LAYER_NUM = args.seqpe_layer_num
    config.SEQPE_LAST_LAYERNORM = args.seqpe_last_layernorm
    config.SEQPE_SCALE_ATTN_WEIGHTS = args.seqpe_scale_attn_weights
    config.SEQPE_ATTN_PDROP = args.seqpe_attn_pdrop
    config.SEQPE_RESID_PDROP = args.seqpe_resid_pdrop
    config.SEQPE_DECAY = args.seqpe_decay
    config.SEQPE_ACTIVATION_FUNCTION = args.seqpe_activation_function
    config.SEQPE_ATTN_DIRECTION = args.seqpe_attn_direction
    config.SEQPE_MASK_PADDING = args.seqpe_mask_padding
    config.SEQPE_ADD_OUT_PROJ = args.seqpe_add_out_proj
    config.SEQPE_PRETRAINED = args.seqpe_pretrained
    config.SEQPE_TEMPERATURE = args.seqpe_temperature
    config.SEQPE_FREEZE_EPOCH_NUM = args.seqpe_freeze_epoch_num
    config.SEQPE_INIT_NORM_WEIGHT = args.seqpe_init_norm_weight
    config.SEQPE_LOGIT_SCALED_LOSS = args.seqpe_logit_scaled_loss

    # seqpe transfer loss config
    config.SEQPE_TRANSFER_WEIGHT = args.seqpe_transfer_weight
    config.SEQPE_TRANSFER_BETA = args.seqpe_transfer_beta
    config.SEQPE_TRANSFER_METRIC = args.seqpe_transfer_metric
    config.SEQPE_TRANSFER_BATCH_SIZE = args.seqpe_transfer_batch_size
    config.SEQPE_TRANSFER_NUM = args.seqpe_transfer_num
    
    # seqpe contrastive loss config
    config.SEQPE_CONTRASTIVE_WEIGHT = args.seqpe_contrastive_weight
    config.SEQPE_CONTRASTIVE_BATCH_SIZE = args.seqpe_contrastive_batch_size
    config.SEQPE_CONTRASTIVE_NUM = args.seqpe_contrastive_num
    config.freeze()

    return config