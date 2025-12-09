from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import random


class SeqPeContrastiveCriterion(_Loss):
    def __init__(self, label_smoothing=0.0, num_heads=12, seqpe_logit_scaled_loss=1.0, seqpe_multi_head_loss=False):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.num_heads = num_heads
        self.seqpe_logit_scaled_loss = seqpe_logit_scaled_loss
        self.seqpe_multi_head_loss = seqpe_multi_head_loss

    def forward(self, pivot_pe, neg_pe, labels):
        batch_size = pivot_pe.size(0)
        n_neg = neg_pe.size(0) // batch_size
        if self.seqpe_multi_head_loss:
            neg_pe = neg_pe.reshape(batch_size, n_neg, self.num_heads, -1).permute(0, 2, 1, 3) # [B, H, N, D]
            pivot_pe = pivot_pe.reshape(batch_size, 1, self.num_heads, -1).permute(0, 2, 1, 3) # [B, 1, H, D]
            logits = pivot_pe @ neg_pe.transpose(-1, -2)
            logits = logits.view(batch_size*self.num_heads, -1)
            labels = labels.unsqueeze(-1).repeat(1, self.num_heads).reshape(-1)
        else:
            neg_pe = neg_pe.reshape(batch_size, n_neg, -1).transpose(1, 2) # [B, H, N]
            pivot_pe = pivot_pe.unsqueeze(1) # [B, 1, H]
            logits = pivot_pe @ neg_pe
            logits = logits.view(batch_size, -1)
        loss = F.cross_entropy(logits * self.seqpe_logit_scaled_loss, labels, label_smoothing=self.label_smoothing)
        return loss


class SeqPeTransferCriterion(_Loss):
    def __init__(self, beta=1.0, metric='kl_div', num_heads=12, seqpe_logit_scaled_loss=1.0, seqpe_multi_head_loss=True):
        """
        beta is ratio for the main transfer loss
        """
        super().__init__()
        self.beta = beta
        self.metric = metric
        self.seqpe_logit_scaled_loss = seqpe_logit_scaled_loss
        self.num_heads = num_heads
        self.seqpe_multi_head_loss = seqpe_multi_head_loss

    def forward(self, main_pe, batch_size, main_indices, trans_pe):
        if main_pe.dim() == 3:
            main_pe = main_pe[0]
        n_trans = main_indices.size(0) // batch_size
        
        if self.seqpe_multi_head_loss:
            trans_pe = trans_pe.reshape(batch_size, n_trans, self.num_heads, -1).permute(0, 2, 1, 3)
            main_pe = main_pe[main_indices].reshape(batch_size, n_trans, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            trans_pe = trans_pe.reshape(batch_size, n_trans, -1)
            main_pe = main_pe[main_indices].reshape(batch_size, n_trans, -1)
        train_dist = F.log_softmax(main_pe @ main_pe.transpose(-1, -2) * self.seqpe_logit_scaled_loss, dim=-1)
        trans_dist = F.log_softmax(trans_pe @ trans_pe.transpose(-1, -2) * self.seqpe_logit_scaled_loss, dim=-1)

        if self.metric == "kl_div":
            loss = self.beta * (
                F.kl_div(input=trans_dist, target=train_dist.detach(), log_target=True) + \
                F.kl_div(input=train_dist.detach(), target=trans_dist, log_target=True)
            )
            # loss = loss + (1-self.beta) * F.kl_div(input=trans_dist.detach(), target=train_dist, log_target=True)
            if self.beta < 1:
                loss = loss + (1-self.beta) * (
                    F.kl_div(input=trans_dist.detach(), target=train_dist, log_target=True) + \
                    F.kl_div(input=train_dist, target=trans_dist.detach(), log_target=True)
                )
        elif self.metric == "mse":
            loss = self.beta * F.mse_loss(input=trans_dist, target=train_dist.detach()) + (1-self.beta) *  F.mse_loss(input=trans_dist.detach(), target=train_dist)
        else:
            raise NotImplementedError
        return loss
        
    