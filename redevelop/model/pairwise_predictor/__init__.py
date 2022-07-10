# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from .pairwise_concat import PairwiseConcat, PairwiseConcatWithResNet, PairwiseConcatWithResNet_ExtraFeat
from .self_attention import SelfAttention

def choose_pairwise_predictor(pairwise_predictor_name, backbone_args, backbone_alphabet):
    """
    :param pairwise_predictor_name:
    :param backbone_args:
    :param backbone_alphabet:
    :return:
    Methods Notes:
    1.Tied Row Attention - Based
    "build-in": the logistic regression based on multi-layer tied row attention (build-in)
    "LR": the logistic regression based on multi-layer tied row attention (external)
    "MultiLayerTiedRowAttention-$lr1,$lr2,...": could select layers and set num_classes additionally
                                                $lr in [1,12]
    2.Embedding - Based
    $reduction in {"first", "mean", "attention"}
    'InnerProduct-$reduction': PairwiseInnerProduct
    'PairwiseConcat-$reduction': PairwiseConcat + Linear
    'SelfAttention-$reduction': SelfAttention
    """
    if pairwise_predictor_name == 'build-in':
        pairwise_predictor = None
        #backbone_control_parameter_dict = {"return_contacts": True, "need_head_weights": True}
    elif 'pairwise-concat' in pairwise_predictor_name:
        pairwise_predictor = PairwiseConcat.create_module_with_name(
            pairwise_predictor_name, backbone_args, backbone_alphabet
        )
    elif 'pc-resnet-with-extra-feat' in pairwise_predictor_name:
        pairwise_predictor = PairwiseConcatWithResNet_ExtraFeat.create_module_with_name(
            pairwise_predictor_name, backbone_args, backbone_alphabet
        )
    elif 'pc-resnet' in pairwise_predictor_name:
        pairwise_predictor = PairwiseConcatWithResNet.create_module_with_name(
            pairwise_predictor_name, backbone_args, backbone_alphabet
        )
    elif 'self-attention' in pairwise_predictor_name:
        pairwise_predictor = SelfAttention.create_module_with_name(
            pairwise_predictor_name, backbone_args, backbone_alphabet
        )
    elif pairwise_predictor_name == "none":
        pairwise_predictor = None
        print("Without Independent Contact Predictor!")
    else:
        raise Exception("Wrong Backbone Type!")

    return pairwise_predictor