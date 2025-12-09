import os
import torch
import torch.distributed as dist
import numpy as np
import pdb
import subprocess
import yaml
import wandb
import argparse


def batched_padded_positions(batch_padding_nums: torch.Tensor, N: int):
    idx = torch.arange(N, device=batch_padding_nums.device).unsqueeze(0)
    positions = idx - batch_padding_nums.unsqueeze(1)
    mask = positions < 0
    positions[mask] = 0
    return positions


def batched_tril_mask(
    N: int,
    diags: torch.Tensor,
    dtype: torch.dtype = torch.long
) -> torch.Tensor:
    B = diags.numel()

    idx = torch.arange(N, device=diags.device)            # [N]
    i = idx.view(1, N, 1)                                 # [1, N, 1]
    j = idx.view(1, 1, N)                                 # [1, 1, N]

    # compare for each batch
    mask = (j - i <= diags.view(B, 1, 1)).to(dtype)                 # [B, N, N]
    return mask

def batched_left_padded_tril_mask(N: int, batch_padding_nums: torch.Tensor):
    assert batch_padding_nums.ndim == 1
    device = batch_padding_nums.device
    row_indices = torch.arange(N, device=device).unsqueeze(1)
    col_indices = torch.arange(N, device=device).unsqueeze(0)
    tril_mask = col_indices <= row_indices

    batch_padding_nums = batch_padding_nums.unsqueeze(-1).unsqueeze(-1)
    col_clear_mask = col_indices >= batch_padding_nums
    combined_mask = tril_mask.unsqueeze(0) & col_clear_mask

    return combined_mask.float()


def greedy_search(model,
                  inputs: dict,
                  pe: torch.Tensor,
                  max_length: int = 256,
                  eos_token_id: int = None,
                  pad_token_id: int = None,
                  device: torch.device = None) -> torch.LongTensor:
    if device is None:
        device = next(model.parameters()).device
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    batch_size, cur_len = input_ids.size()
    max_length = max(max_length, cur_len * 2)
    generated = input_ids.clone()
    is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        for step in range(max_length - cur_len):
            model_inputs = {'input_ids': generated}
            if attention_mask is not None:
                model_inputs['attention_mask'] = attention_mask
            
            if pe is not None:
                model_inputs['pe'] = pe

            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if pad_token_id is not None:
                next_tokens = torch.where(is_finished.unsqueeze(-1),
                                          torch.full_like(next_tokens, pad_token_id),
                                          next_tokens)

            generated = torch.cat([generated, next_tokens], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
                    dim=-1
                )

            if eos_token_id is not None:
                is_finished = is_finished | (next_tokens.squeeze(-1) == eos_token_id)
                if is_finished.all():
                    break

        # Optionally pad sequences to max_length
        if pad_token_id is not None and generated.size(1) < max_length:
            padding_length = max_length - generated.size(1)
            pad_tensor = torch.full((batch_size, padding_length), pad_token_id, device=device, dtype=generated.dtype)
            generated = torch.cat([generated, pad_tensor], dim=-1)

    return generated



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_git_commit_hash():
    try:
        # Run the git command and capture its output
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).strip()
        # Decode from bytes to string
        return commit_hash.decode("utf-8")
    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., not a git repository)
        return f"-1"
    except FileNotFoundError:
        # Handle the case where git is not installed
        return "-2"


def get_slurm_job_info():
    job_id = os.getenv("SLURM_JOB_ID")  # The unique job ID
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")  # The array job ID (same for all)
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")  # The specific task ID
    return {"SLURM_JOB_ID": job_id, "SLURM_ARRAY_JOB_ID": array_job_id, "SLURM_ARRAY_TASK_ID": task_id}


def init_wandb(config, args, proj_name, run_name=''):
    api_key = os.getenv("WANDB_API_KEY")
    job_info = get_slurm_job_info()
    config_json = yaml.safe_load(config.dump())
    config_json['args'] = {}
    for k, v in vars(args).items():
        config_json['args'][k] = v
    config_json['CODE_VERSION'] = get_git_commit_hash()
    config_json['SLURM_JOB_INFO'] = job_info
    wandb.login(key=api_key)
    wandb.init(
        # set the wandb project where this run will be logged
        project=proj_name,
        # track hyperparameters and run metadata
        config=config_json,
        name=run_name if run_name.strip() else None
    )


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm