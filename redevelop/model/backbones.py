# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import fm

def choose_backbone(backbone_name):
    # 1.ESM1b
    if backbone_name == 'esm1b':
        backbone, backbone_alphabet = fm.pretrained.esm1b_t33_650M_UR50S()
        #backbone_batch_converter = backbone_alphabet.get_batch_converter()
    elif backbone_name == 'esm1b-rna':
        backbone, backbone_alphabet = fm.pretrained.esm1b_rna_t12()

    elif backbone_name == 'esm1':
        backbone, backbone_alphabet = fm.pretrained.esm1_t6_43M_UR50S()

    # 2.MSA Transformer
    elif backbone_name == 'msa_transformer':
        backbone, backbone_alphabet = fm.pretrained.esm_msa1_t12_100M_UR50S()
    else:
        raise Exception("Wrong Backbone Type! {}".format(backbone_name))

    return backbone, backbone_alphabet