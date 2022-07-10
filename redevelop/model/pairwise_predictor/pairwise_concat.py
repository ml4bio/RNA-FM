import torch
from torch import nn
from ..downstream_module import DownStreamModule

from collections import OrderedDict

class PairwiseConcat(DownStreamModule):
    """
    contact predictor with pairwise concat
    """
    def __init__(self, backbone_args, backbone_alphabet, num_classes,
                 symmetric=False, embed_reduction=-1, depth_reduction="mean"):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super().__init__(backbone_args, backbone_alphabet, depth_reduction,
                         need_token=False, need_attention=[], need_embedding=[12], need_extrafeat=[])
        self.embed_dim_in = self.backbone_args.embed_dim
        self.attention_heads = self.backbone_args.attention_heads

        self.num_classes = num_classes
        self.symmetric = symmetric

        self.embed_reduction = embed_reduction
        if self.embed_reduction != -1:
            self.embed_dim_out = self.embed_reduction
            self.pre_reduction = nn.Linear(self.embed_dim_in, self.embed_dim_out)
        else:
            self.embed_dim_out = self.embed_dim_in
            self.pre_reduction = None

        self.proj = nn.Conv2d(self.embed_dim_out * 2, self.num_classes, kernel_size=1)



    def forward(self, tokens, inputs):
        embeddings = inputs["embedding"]
        # remove auxiliary tokens
        embeddings, padding_masks = self.remove_pend_tokens_1d(tokens, embeddings)

        if len(embeddings.size()) == 3:       # for seq
            batch_size, seqlen, hiddendim = embeddings.size()                    # B, L, E
        elif len(embeddings.size()) == 4:     # for msa
            batch_size, depth, seqlen, hiddendim = embeddings.size()
            embeddings = self.msa_depth_reduction(embeddings, padding_masks)     # B, L, E
        else:
            raise Exception("Unknown Embedding Type!")

        # embedding dim reduction
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(embeddings)
            hiddendim = self.embed_reduction

        # inner product
        # embedding_T = embedding.permute(0, 2, 1)
        # inner_product_map = torch.matmul(embedding, embedding_T)
        # contact_map = self.activation(inner_product_map)
        # with BN
        # inner_product_map = torch.matmul(embedding, embedding_T).unsqueeze(1)
        # inner_product_map = self.activation(self.BN(inner_product_map)).squeeze(1)

        # cosine similarity
        embeddings = embeddings.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        embedding_T = embeddings.permute(0, 2, 1, 3)
        pairwise_concat_embedding = torch.cat([embeddings, embedding_T], dim=3)
        pairwise_concat_embedding = pairwise_concat_embedding.permute(0, 3, 1, 2)

        output = self.proj(pairwise_concat_embedding)

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


class PairwiseConcatWithResNet(PairwiseConcat):
    """
    contact predictor with pairwise concat + resnet
    reproduce from msa tranformer
    """
    def __init__(self, backbone_args, backbone_alphabet, num_classes,
                 symmetric=False, embed_reduction=128, depth_reduction="mean"):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super().__init__(backbone_args, backbone_alphabet, num_classes, symmetric, embed_reduction, depth_reduction,)

        self.embed_reduction = embed_reduction

        first_layer = nn.Sequential(
            nn.Conv2d(self.embed_reduction*2, 64, kernel_size=1),
        )

        self.num_res_layers = 32
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=64, planes=64, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        final_layer = nn.Conv2d(64, 2, kernel_size=3, padding=1)

        layers = OrderedDict()
        layers["first"] = first_layer
        layers["resnet"] = res_layers
        layers["final"] = final_layer

        self.proj = nn.Sequential(layers)


class PairwiseConcatWithResNet_ExtraFeat(PairwiseConcat):
    """
    contact predictor with pairwise concat + resnet
    reproduce from msa tranformer
    """

    def __init__(self, backbone_args, backbone_alphabet, num_classes,
                 symmetric=False, embed_reduction=1, depth_reduction="mean", extra_feats=[], with_embbedding=True):
        """
        :param depth_reduction: mean, first, (adaptive)
        :param extra_feats: "msa-cov", "petfold-ss"
        """
        super().__init__(backbone_args, backbone_alphabet, num_classes, symmetric, embed_reduction, depth_reduction,)

        # resnet32
        self.extra_feats = extra_feats
        self.with_embedding = True if int(with_embbedding) == 1 else False
        if self.extra_feats is not None:
            self.input_type["extra-feat"] = self.extra_feats
            self.extra_feat_dim = 0
            for key in self.extra_feats:
                if key == "msa-cov":
                    self.extra_feat_dim += 25
                elif key == "petfold-ss":
                    self.extra_feat_dim += 1

        main_filter_num = 64
        if self.with_embedding == True:
            first_layer = nn.Sequential(
                nn.Conv2d(self.embed_reduction*2 + self.extra_feat_dim, main_filter_num, kernel_size=1),
            )
        else:
            first_layer = nn.Sequential(
                nn.Conv2d(self.extra_feat_dim, main_filter_num, kernel_size=1),
            )


        self.num_res_layers = 32
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=main_filter_num, planes=main_filter_num, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        final_layer = nn.Conv2d(main_filter_num, 2, kernel_size=3, padding=1)

        layers = OrderedDict()
        layers["first"] = first_layer
        layers["resnet"] = res_layers
        layers["final"] = final_layer

        self.proj = nn.Sequential(layers)

    def forward(self, tokens, inputs):
        embeddings = inputs["embedding"]
        # remove auxiliary tokens
        embeddings, padding_masks = self.remove_pend_tokens_1d(tokens, embeddings)

        if len(embeddings.size()) == 3:       # for seq
            batch_size, seqlen, hiddendim = embeddings.size()                    # B, L, E
        elif len(embeddings.size()) == 4:     # for msa
            batch_size, depth, seqlen, hiddendim = embeddings.size()
            embeddings = self.msa_depth_reduction(embeddings, padding_masks)     # B, L, E
        else:
            raise Exception("Unknown Embedding Type!")

        # embedding dim reduction
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(embeddings)
            hiddendim = self.embed_reduction

        # inner product
        # embedding_T = embedding.permute(0, 2, 1)
        # inner_product_map = torch.matmul(embedding, embedding_T)
        # contact_map = self.activation(inner_product_map)
        # with BN
        # inner_product_map = torch.matmul(embedding, embedding_T).unsqueeze(1)
        # inner_product_map = self.activation(self.BN(inner_product_map)).squeeze(1)

        # cosine similarity
        embeddings = embeddings.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        embedding_T = embeddings.permute(0, 2, 1, 3)
        pairwise_concat_embedding = torch.cat([embeddings, embedding_T], dim=3)
        pairwise_concat_embedding = pairwise_concat_embedding.permute(0, 3, 1, 2)

        # extra-feat
        extra_feat_matrix = []
        for key in self.extra_feats:
            extra_feat_matrix.append(inputs[key].permute(0, 3, 1, 2))
        extra_feat_matrix = torch.cat(extra_feat_matrix, dim=1)
        if self.with_embedding == True:
            pairwise_concat_embedding = torch.cat([pairwise_concat_embedding, extra_feat_matrix], dim=1)
        else:
            pairwise_concat_embedding = extra_feat_matrix

        output = self.proj(pairwise_concat_embedding)

        if self.symmetric == True:
            upper_triangular_output = torch.triu(output)
            lower_triangular_output = torch.triu(output, diagonal=1).permute(0, 1, 3, 2)
            output = upper_triangular_output + lower_triangular_output

        return output

    @classmethod
    def create_module_with_name(cls, module_name, backbone_args, backbone_alphabet):
        _, num_class, symmetric, depth_reduction, extra_feats, with_embbedding = module_name.split("_")
        num_class = int(num_class)
        if symmetric == "sym":
            symmetric = True
        elif symmetric == "asym":
            symmetric = False
        else:
            raise Exception("Wrong Symmetric Type!")
        extra_feats = extra_feats.split("+")
        module = cls(backbone_args, backbone_alphabet, num_classes=num_class, symmetric=symmetric,
                     depth_reduction=depth_reduction, extra_feats=extra_feats, with_embbedding=with_embbedding)
        return module


# Res Block
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=dilation, groups=groups, bias=False, dilation=dilation)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MyBasicResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MyBasicResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # cjy commented
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu2(out)
        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.relu(out)

        return out
