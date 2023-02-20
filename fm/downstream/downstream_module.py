# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""
from torch import nn
import torch
from functools import wraps




class DownStreamModule(nn.Module):
    """
    base contact predictor for msa
    """
    def __init__(self, backbone_args, backbone_alphabet, depth_reduction="none",
                 need_token=False, need_attention=[], need_embedding=[12], need_extrafeat=[]):
        super().__init__()
        self.backbone_args = backbone_args
        self.backbone_alphabet = backbone_alphabet

        self.prepend_bos = self.backbone_alphabet.prepend_bos
        self.append_eos = self.backbone_alphabet.append_eos
        self.bos_idx = self.backbone_alphabet.cls_idx
        self.eos_idx = self.backbone_alphabet.eos_idx
        if self.append_eos and self.eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.pad_idx = self.backbone_alphabet.padding_idx

        self.embed_dim = self.backbone_args.embed_dim
        self.attention_heads = self.backbone_args.attention_heads

        self.depth_reduction = depth_reduction
        if self.depth_reduction == "attention":
            self.msa_embed_dim_in = self.embed_dim
            self.msa_embed_dim_out = self.embed_dim // self.attention_heads
            self.msa_q_proj = nn.Linear(self.msa_embed_dim_in, self.msa_embed_dim_out)
            self.msa_k_proj = nn.Linear(self.msa_embed_dim_in, self.msa_embed_dim_out)

        self.input_type = {
            "token": need_token,
            "attention": need_attention,
            "embedding": need_embedding,
            "extra-feat": need_extrafeat,
        }


    def remove_pend_tokens_1d(self, tokens, seqs):
        """
        :param tokens:
        :param seqs: must be shape of [B, ..., L, E]    # seq: [B, L, E]; msa: [B, D, L, E]
        :return:
        """
        padding_masks = tokens.ne(self.pad_idx)

        # remove eos token  （suffix first）
        if self.append_eos:     # default is right
            eos_masks = tokens.ne(self.eos_idx)
            eos_pad_masks = (eos_masks & padding_masks).to(seqs)
            seqs = seqs * eos_pad_masks.unsqueeze(-1)
            seqs = seqs[:, ..., :-1, :]
            padding_masks = padding_masks[:, ..., :-1]

        # remove bos token
        if self.prepend_bos:    # default is left
            seqs = seqs[:, ..., 1:, :]
            padding_masks = padding_masks[:, ..., 1:]

        if not padding_masks.any():
            padding_masks = None

        return seqs, padding_masks

    def remove_pend_tokens_2d(self, tokens, maps):
        """
        :param tokens:
        :param maps: shape of [B, L, L, ...]
        :return:
        """
        padding_masks = tokens.ne(self.pad_idx)

        # remove eos token  （suffix first）
        if self.append_eos:  # default is right
            eos_masks = tokens.ne(self.eos_idx)
            eos_pad_masks = (eos_masks & padding_masks).to(maps)
            eos_pad_masks = eos_pad_masks.unsqueeze(1) * eos_pad_masks.unsqueeze(2)
            maps = maps * eos_pad_masks.unsqueeze(-1)
            maps = maps[:, :-1, :-1, ...]
            padding_masks = padding_masks[:, :-1, ...]

        # remove bos token
        if self.prepend_bos:  # default is left
            maps = maps[:, 1:, 1:, ...]
            padding_masks = padding_masks[:, 1:, ...]

        if not padding_masks.any():
            padding_masks = None

        return maps, padding_masks

    def msa_depth_reduction(self, embeddings, padding_masks):
        """
        :param embeddings:  B,
        :param padding_masks:
        :return:
        """
        if self.depth_reduction == "first":
            embeddings = embeddings[:, 0, :, :]
        elif self.depth_reduction == "mean":
            embeddings = torch.mean(embeddings, dim=1)
        elif self.depth_reduction == "attention":
            msa_q = self.msa_q_proj(embeddings[:, 0, :, :])  # first query
            msa_k = self.msa_k_proj(embeddings)  # all keys
            if padding_masks is not None:
                # Zero out any padded aligned positions - this is important since
                # we take a sum across the alignment axis.
                msa_q = msa_q * (1 - padding_masks[:, 0, :].unsqueeze(-1).type_as(msa_q))
            depth_attn_weights = torch.einsum("bld,bjld->bj", msa_q, msa_k)
            depth_attn_weights = torch.softmax(depth_attn_weights, dim=1)
            embeddings = torch.sum(embeddings * depth_attn_weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        else:
            raise Exception("Wrong Depth Reduction Type")

        return embeddings