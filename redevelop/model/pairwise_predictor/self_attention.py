import torch
from torch import nn
from ..downstream_module import DownStreamModule

class SelfAttention(DownStreamModule):
    """
    contact predictor with attention
    """
    def __init__(self, backbone_args, backbone_alphabet, num_classes, symmetric=False, depth_reduction="mean"):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super().__init__(backbone_args, backbone_alphabet, depth_reduction,
                         need_token=False, need_attention=[], need_embedding=[12])
        self.embed_dim_in = self.backbone_args.embed_dim
        self.attention_heads = self.backbone_args.attention_heads

        self.num_classes = num_classes
        self.symmetric = symmetric

        self.class_embed_dim_out = 256

        self.q_proj = nn.Linear(self.embed_dim_in, self.num_classes * self.class_embed_dim_out)
        self.k_proj = nn.Linear(self.embed_dim_in, self.num_classes * self.class_embed_dim_out)

    def forward(self, tokens, inputs):
        embeddings = inputs["embedding"]
        # remove auxiliary tokens
        embeddings, padding_masks = self.remove_pend_tokens_1d(tokens, embeddings)

        if len(embeddings.size()) == 3:       # for seq
            batch_size, seqlen, hiddendim = embeddings.size()
        elif len(embeddings.size()) == 4:     # for msa
            batch_size, depth, seqlen, hiddendim = embeddings.size()
            embeddings = self.msa_depth_reduction(embeddings, padding_masks)
        else:
            raise Exception("Unknown Embedding Type!")

        # attention
        q = self.q_proj(embeddings).view(batch_size, seqlen, self.num_classes, self.class_embed_dim_out)
        k = self.k_proj(embeddings).view(batch_size, seqlen, self.num_classes, self.class_embed_dim_out)

        # short-cut
        #embeddings = embeddings.unsqueeze(-2).expand_as(q)
        #q = embeddings + q
        #k = embeddings + k

        output = torch.einsum("bick,bjck->bcij", q, k)

        if self.symmetric == True:
            upper_triangular_output = torch.triu(output)
            lower_triangular_output = torch.triu(output, diagonal=1).permute(0, 1, 3, 2)
            output = upper_triangular_output + lower_triangular_output

        return output

    @classmethod
    def create_module_with_name(cls, module_name, backbone_args, backbone_alphabet):
        _, num_class, symmetric, depth_reduction = module_name.split("_")
        num_class = int(num_class)
        if symmetric == "sym":
            symmetric = True
        elif symmetric == "asym":
            symmetric = False
        else:
            raise Exception("Wrong Symmetric Type!")
        module = cls(backbone_args, backbone_alphabet,
                     num_classes=num_class, symmetric=symmetric, depth_reduction=depth_reduction)
        return module



