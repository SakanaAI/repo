from .sequential_pe import (
    SequentialPE, SinusoidalPositionalEmbedding, RoFormerSinusoidalPositionalEmbedding,
    RoFormer2DPositionalEmbedding, RoFormer2DMixedPositionalEmbedding, AlibiPositionalEmbedding,
    TableEmbedding
)
from .pe_sampler import SeqPeMainTaskPosSampler, SeqPeContrstiveDataSampler, SeqPeTransferDataSampler
from .pe_criterion import SeqPeContrastiveCriterion, SeqPeTransferCriterion
from .pe_utils import DynamicPeMode
from functools import partial


def build_eval_pe_sampler(config, input_shape, add_cls_token_pe=False, start_pos=None):
    pe_main_task_sampler = SeqPeMainTaskPosSampler(
        input_shape, config.PE_MAX_POSITION, config.SEQPE_MAX_DIGITS, data_dim=config.PE_DATA_DIM,
        add_cls_token_pe=add_cls_token_pe, use_random_shift=False, default_start_pos=start_pos
    ) if config.PE_TYPE != "vanilla" else None
    return pe_main_task_sampler


def build_pe_sampler_and_criterion(config, input_shape, train_data_size, args, add_cls_token_pe=False, start_epoch=0, seed=None):

    ####################################################
    # prepare pe data samplers
    """
    NOTE: explanation for add_cls_token_pe
    add_cls_token_pe=Ture: the PE model will generate the PE of cls.
    add_cls_token_pe=Ture: we use a special pos embedding to model the pos of cls.

    This is because that seq_pe is hard to represent cls's pos in 2d situation.
    """
    use_random_shift = config.PE_USE_RANDOM_SHIFT if hasattr(config, 'PE_USE_RANDOM_SHIFT') else False
    pe_main_task_sampler = SeqPeMainTaskPosSampler(
        input_shape, config.PE_MAX_POSITION, config.SEQPE_MAX_DIGITS, data_dim=config.PE_DATA_DIM,
        add_cls_token_pe=add_cls_token_pe, use_random_shift=use_random_shift,
        random_shift_rate=config.PE_RANDOM_SHIFT_RATE, random_shift_downsample=config.PE_RANDOM_SHIFT_DOWNSAMPLE,
        start_epoch=start_epoch
    ) if config.PE_TYPE not in ["vanilla", "nope"] else None
    data_size_multiplier = args.seqpe_data_size_multiplier if hasattr(args, 'seqpe_data_size_multiplier') else 1 # intuitively set to increase data diversity by creating more random data
    pe_ct_sampler = SeqPeContrstiveDataSampler(
        train_data_size * data_size_multiplier, config.SEQPE_CONTRASTIVE_BATCH_SIZE,
        config.PE_MAX_POSITION, config.SEQPE_MAX_DIGITS, config.SEQPE_CONTRASTIVE_NUM, data_dim=config.PE_DATA_DIM,
        distributional_sample_range=config.SEQPE_DIST_SAMPLE_RANGE, start_epoch=start_epoch, seed=seed
    ) if config.PE_TYPE == "seq_pe" else None
    pe_trans_sampler = SeqPeTransferDataSampler(
        input_shape, train_data_size * data_size_multiplier, config.SEQPE_TRANSFER_BATCH_SIZE, config.PE_MAX_POSITION,
        config.SEQPE_MAX_DIGITS, transfer_size=config.SEQPE_TRANSFER_NUM, data_dim=config.PE_DATA_DIM,
        start_epoch=start_epoch
    ) if config.PE_TYPE == "seq_pe" else None
    
    # seqpe_multi_head_loss = config.SEQPE_MULTI_HEAD_LOSS if config.USE_PE_MULTI_HEAD else False
    seqpe_logit_scaled_loss=config.SEQPE_LOGIT_SCALED_LOSS
    num_heads = config.NUM_ATTENTION_HEADS
    pe_ct_criterion = SeqPeContrastiveCriterion() if config.PE_TYPE == "seq_pe" else None
    pe_trans_criterion = SeqPeTransferCriterion(config.SEQPE_TRANSFER_BETA, config.SEQPE_TRANSFER_METRIC,num_heads=num_heads, seqpe_logit_scaled_loss=seqpe_logit_scaled_loss, seqpe_multi_head_loss=True) if config.PE_TYPE == "seq_pe" else None
    return pe_main_task_sampler, pe_ct_sampler, pe_trans_sampler, pe_ct_criterion, pe_trans_criterion


def build_pe_model(config):
    pe_type = config.PE_TYPE
    pe_apply_method = config.PE_APPLY_METHOD

    if pe_type == "seq_pe":
        pe_model = SequentialPE(
            pe_embed_dim=config.PE_EMBED_DIM,
            num_attention_heads=config.NUM_ATTENTION_HEADS,
            layer_num=config.SEQPE_LAYER_NUM,
            max_digits=config.SEQPE_MAX_DIGITS,
            attn_direction=config.SEQPE_ATTN_DIRECTION,
            use_last_layernorm=config.SEQPE_LAST_LAYERNORM,
            mask_padding=config.SEQPE_MASK_PADDING,
            activation_function=config.SEQPE_ACTIVATION_FUNCTION,
            resid_pdrop=config.SEQPE_RESID_PDROP,
            attn_pdrop=config.SEQPE_ATTN_PDROP,
            scale_attn_weights=config.SEQPE_SCALE_ATTN_WEIGHTS,
            use_cls_token=True,
            out_proj_dim=config.PE_OUT_PROJ_DIM,
            seqpe_temperature=config.SEQPE_TEMPERATURE,
            seqpe_init_norm_weight=config.SEQPE_INIT_NORM_WEIGHT if hasattr(config, 'SEQPE_INIT_NORM_WEIGHT') else 1.0,
            data_dim=config.PE_DATA_DIM
        )
    elif pe_type == "table_pe":
        if config.TABLEPE_PRETRAINED:
            pe_model = TableEmbedding.from_pretrained(
                pretrained_path=config.TABLEPE_PRETRAINED,
                freeze_pretrained_pe=config.FREEZE_TABLEPE
            )
        else:
            pe_model = TableEmbedding(
                num_positions=config.PE_MAX_POSITION + 1,
                embedding_dim=config.PE_OUT_PROJ_DIM,
            )
    elif pe_type == "sin":
        # +1 is for the cls token
        pe_model = SinusoidalPositionalEmbedding(
            num_positions=config.PE_MAX_POSITION + 1,
            embedding_dim=config.PE_OUT_PROJ_DIM,
            base=config.SINUSOIDAL_PE_BASE,
            init_scale=config.SINUSOIDAL_PE_SCALE if hasattr(config, "SINUSOIDAL_PE_SCALE") else 1.0,
        )
    elif pe_type == "rotary":
        # +1 is for the cls token
        if pe_apply_method == "rotary":
            pe_model = RoFormerSinusoidalPositionalEmbedding(
                num_positions=config.PE_MAX_POSITION + 1,
                embedding_dim=config.PE_OUT_PROJ_DIM,
                base=config.SINUSOIDAL_PE_BASE
            )
        # NOTE: rotary_2d and rotary_mixed are for 2D data
        # elif pe_apply_method == "rotary_2d":
        #     pe_model = RoFormer2DPositionalEmbedding(
        #         num_positions=config.PE_MAX_POSITION,
        #         embedding_dim=config.PE_OUT_PROJ_DIM,
        #         base=config.SINUSOIDAL_PE_BASE                
        #     )
        # elif pe_apply_method == "rotary_mixed":
        #     pe_model = RoFormer2DMixedPositionalEmbedding(
        #         num_positions=config.PE_MAX_POSITION,
        #         embedding_dim=config.PE_OUT_PROJ_DIM,
        #         num_heads=config.MODEL.NUM_HEADS,
        #         vit_depth=config.MODEL.DEPTH,
        #         base=config.SINUSOIDAL_PE_BASE
        #     )
        else:
            raise NotImplementedError
    elif pe_type in ["vanilla", "nope"]:
        pe_model = None
    elif pe_type == "alibi":
        pe_model = AlibiPositionalEmbedding(
            num_positions=config.PE_MAX_POSITION,
            num_heads=config.NUM_ATTENTION_HEADS,
        )
    else:
        raise NotImplementedError
    return pe_model


def build_pe_gen_func(config):
    pe_type = config.PE_TYPE
    base=config.SINUSOIDAL_PE_BASE
    dim = config.PE_OUT_PROJ_DIM
    pe_gen_func = None
    dynamic_pe_mode = config.DYNAMIC_PE_MODE if hasattr(config, 'DYNAMIC_PE_MODE') else DynamicPeMode.N
    if dynamic_pe_mode in [DynamicPeMode.C, DynamicPeMode.CWSC, DynamicPeMode.A]:
        if pe_type == "rotary":
            pe_gen_func = partial(RoFormerSinusoidalPositionalEmbedding.from_positions, dim=dim, base=base)
        elif pe_type == "sin":
            pe_gen_func = partial(SinusoidalPositionalEmbedding.from_positions, dim=dim, base=base)
        elif pe_type == "alibi":
            pe_gen_func = partial(AlibiPositionalEmbedding.from_positions, num_heads=config.NUM_ATTENTION_HEADS)
    return pe_gen_func