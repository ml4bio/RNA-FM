# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fm
import torch
from argparse import Namespace
import warnings
import urllib
from pathlib import Path
import os

def load_model_and_alphabet(model_name):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name)
    else:
        return load_model_and_alphabet_hub(model_name)

def load_hub_workaround(url, download_name=None):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=True, map_location='cpu', file_name=download_name)
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        if download_name == None:
            fn = Path(url).name
        else:
            fn = download_name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    return data



def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data

def load_model_and_alphabet_hub(model_name, theme="protein"):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    regression_data = load_regression_hub(model_name)
    return load_model_and_alphabet_core(model_data, regression_data, theme)

def load_model_and_alphabet_local(model_location, theme="protein"):
    """ Load from local path. The regression weights need to be co-located """
    model_data = torch.load(model_location, map_location='cpu')
    try:
        regression_location = model_location[:-3] + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location='cpu')
    except FileNotFoundError:
        regression_data = None
    return load_model_and_alphabet_core(model_data, regression_data, theme)

def load_model_and_alphabet_core(model_data, regression_data=None, theme="protein"):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    alphabet = fm.Alphabet.from_architecture(model_data["args"].arch, theme=theme)

    if model_data["args"].arch == 'roberta_large':
        # upgrade state dict
        pra = lambda s: ''.join(s.split('encoder_')[1:] if 'encoder' in s else s)
        prs1 = lambda s: ''.join(s.split('encoder.')[1:] if 'encoder' in s else s)
        prs2 = lambda s: ''.join(s.split('sentence_encoder.')[1:] if 'sentence_encoder' in s else s)
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
        model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop
        model_type = fm.RNABertModel

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args), alphabet,
    )

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        if expected_missing - found_keys:
            warnings.warn("Regression weights not found, predicting contacts will not produce correct results.")

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet


def rna_fm_t12(model_location=None):
    if model_location is not None and os.path.exists(model_location):
        # local
        return load_model_and_alphabet_local(model_location, theme="rna")  # "./pretrained/RNA-FM_pretrained.pth"
    else:
        return load_rnafm_model_and_alphabet_hub("rna_fm_t12", theme="rna")


def load_rnafm_model_and_alphabet_hub(model_name, theme="rna"):
    if model_name == "rna_fm_t12":
        url = f"https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth"
        model_data = load_hub_workaround(url, download_name="RNA-FM_pretrained.pth")
        #url = f"https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_SS-ResNet.pth"
        #model_data = load_hub_workaround(url, download_name="RNA-FM_SS-ResNet.pth")
        regression_data = None
    else:
        raise Exception("Unknown model name: {}".format(model_name))
    return load_model_and_alphabet_core(model_data, regression_data, theme)

