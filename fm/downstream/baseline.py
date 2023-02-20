# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import torch

from .backbones import choose_backbone
from .pairwise_predictor import choose_pairwise_predictor

from .weights_init import *
from ptflops import get_model_complexity_info
from torch.cuda.amp import autocast

class Baseline(nn.Module):
    def __init__(self,
                 backbone_name,
                 seqwise_predictor_name="none",
                 elewise_predictor_name="none",
                 pairwise_predictor_name="none",
                 backbone_frozen=0,
                 backbone_random_initialization=False,   # because most of models used in this pipeline are pretrained
                 ):
        super(Baseline, self).__init__()
        # 0.Configuration
        self.backbone_name = backbone_name

        self.seqwise_predictor_name = seqwise_predictor_name
        self.elewise_predictor_name = elewise_predictor_name
        self.pairwise_predictor_name = pairwise_predictor_name

        self.backbone_frozen = backbone_frozen
        self.backbone_frozen_output_cache = {}

        self.backbone_random_initialization = backbone_random_initialization

        # just for reference
        self.target_info = {
            "p-ss": {"num_class": 3, "symmetric": None},        # or 8
            "p-contact": {"num_class": 2, "symmetric": True},
            "p-dist-bin": {"num_class": 37, "symmetric": True},
            "p-omega-bin": {"num_class": 25, "symmetric": True},
            "p-phi-bin": {"num_class": 13, "symmetric": False},
            "p-theta-bin": {"num_class": 25, "symmetric": False},
            "r-ss": {"num_class": 2, "symmetric": True},
            "ts-rgs": {"num_class": -1, "symmetric": None},
            "ts-cls": {"num_class": 2, "symmetric": None},
        }

        # 1.Build backbone
        self.backbone, self.backbone_alphabet = choose_backbone(self.backbone_name)
        self.backbone.lm_head.requires_grad_(False)

        self.backbone_control_info = {"return_contacts": False, "need_head_weights": False, "repr_layers": []}

        # 2.Build downstream modules
        self.downstream_modules = nn.ModuleDict()

        # the conposition of each name are "predictor_type" + ":" + "target_name"
        if self.pairwise_predictor_name != "none":
            for pp_name in self.pairwise_predictor_name.split(" "):
                predictor_type, target_name = pp_name.split(":")
                self.downstream_modules[pp_name] = choose_pairwise_predictor(
                    predictor_type, self.backbone.args, self.backbone_alphabet
                )
                self.update_backbone_contral_info(self.downstream_modules[pp_name].input_type)

        # to generate embeddings
        if self.seqwise_predictor_name == "none" and self.elewise_predictor_name == "none" and self.pairwise_predictor_name == "none":
            self.backbone_control_info["repr_layers"] = [12]

        # Initialization of parameters
        # omit backbone initialization
        if self.backbone_random_initialization == True:
            self.backbone.apply(weights_init_kaiming)
            #self.backbone.apply(weights_init_classifier)
        for key in self.downstream_modules.keys():
            self.downstream_modules[key].apply(weights_init_kaiming)
            self.downstream_modules[key].apply(weights_init_classifier)   # 最优 reduction这块还挺重要的

    #@autocast()
    def forward(self, data):
        x = data["token"]
        need_head_weights = self.backbone_control_info["need_head_weights"]
        repr_layers = self.backbone_control_info["repr_layers"]
        return_contacts = self.backbone_control_info["return_contacts"]

        if self.backbone_frozen != -1:
            if self.backbone_frozen == 1:
                self.backbone.eval()
                with torch.no_grad():
                    results = self.backbone(x, need_head_weights=need_head_weights,
                                            repr_layers=repr_layers, return_contacts=return_contacts)
            elif self.backbone_frozen == 2:
                need_forward = False
                for des in data["description"]:
                    if des not in self.backbone_frozen_output_cache:
                        need_forward = True

                if need_forward == True:
                    self.backbone.eval()
                    with torch.no_grad():
                        results = self.backbone(x, need_head_weights=need_head_weights,
                                                repr_layers=repr_layers, return_contacts=return_contacts)
                    self.save_backbone_output_cache(data, results)
                else:
                    results = self.load_backbone_output_cache(data)
            else:  # 0
                results = self.backbone(x, need_head_weights=need_head_weights,
                                        repr_layers=repr_layers, return_contacts=return_contacts)
        else:
            results = {}

        for key in self.downstream_modules.keys():
            key_info = key.split(":")
            predictor_type, target_name = key_info[0], key_info[1]
            ds_module_input = self.fetch_ds_module_input(data, results, self.downstream_modules[key].input_type)
            results[target_name] = self.downstream_modules[key](x, ds_module_input).float()    # .float() for fp16 to fp32

        return results


    def update_backbone_contral_info(self, ds_module_input_type):
        """
        :param ds_module_input_type: the downstream modules may need different output from backbone. This is a dict of
        control parameters for backbone.
        :return: update self.backbone_control_info
        """
        for key in ds_module_input_type.keys():
            if key == "token":
                pass
            elif key == "attention":
                if ds_module_input_type[key] != []:
                    self.backbone_control_info["need_head_weights"] = True
            elif key == "embedding":
                if isinstance(ds_module_input_type[key], int):
                    layer_i = ds_module_input_type[key]
                    if layer_i not in self.backbone_control_info["repr_layers"]:
                        self.backbone_control_info["repr_layers"].append(ds_module_input_type[key])
                elif isinstance(ds_module_input_type[key], list) and len(ds_module_input_type[key]) == 1:
                    layer_i = ds_module_input_type[key][0]
                    if layer_i not in self.backbone_control_info["repr_layers"]:
                        self.backbone_control_info["repr_layers"].append(layer_i)
                else:
                    for layer_i in ds_module_input_type[key]:
                        if layer_i not in self.backbone_control_info["repr_layers"]:
                            self.backbone_control_info["repr_layers"].append(ds_module_input_type[key])
            elif key == "extra-feat":  # features from third parties
                pass
            else:
                raise Exception("Unknown Keys for DS Module Input")


    def fetch_ds_module_input(self, data, backbone_results, ds_module_input_type):
        """
        The input of a specific downstream module may be a subset of backbone's output and we should pick them up into a
        sub_results dictionary and pass it into the downstream stask.
        :param tokens: output of this ds_input, there must be tokens as the first parameter for generate mask.
        :param backbone_results:
        :param ds_module_input_type:
        :return:
        """
        tokens = data["token"]
        ds_input = {}
        for key in ds_module_input_type.keys():
            if key == "token":
                if ds_module_input_type[key] == True:
                    ds_input[key] = tokens
            elif key == "attention":
                if ds_module_input_type[key] != []:
                    attention = backbone_results["attentions"] if "attentions" in backbone_results else backbone_results["row_attentions"]
                    ds_input[key] = attention
            elif key == "embedding":
                if isinstance(ds_module_input_type[key], int):
                    ds_input[key] = backbone_results["representations"][ds_module_input_type[key]]
                elif isinstance(ds_module_input_type[key], list) and len(ds_module_input_type[key]) == 1:
                    ds_input[key] = backbone_results["representations"][ds_module_input_type[key][0]]
                else:
                    for layer_i in ds_module_input_type[key]:
                        ds_input[key] = backbone_results["representations"][layer_i]
            elif key == "extra-feat":  # features from third parties
                for k in ds_module_input_type[key]:
                    ds_input[k] = data[k]
            else:
                raise Exception("Unknown Keys for DS Module Input")
        return ds_input

    def save_backbone_output_cache(self, data, results):
        name_list = data["description"]
        # save cache
        for i, descritption in enumerate(name_list):
            temp_results = {}
            for key in results.keys():
                if isinstance(results[key], dict):
                    temp_results[key] = {}
                    for sub_key in results[key].keys():
                        if key == "representations" and len(results[key][sub_key][i].shape) == 2:
                            # we should save tensor without padding (reserve bos & eos)
                            lrange = (0, self.backbone_alphabet.prepend_bos + data["length"][i] + self.backbone_alphabet.append_eos)
                            temp_results[key][sub_key] = results[key][sub_key][i][lrange[0]:lrange[1]].detach().cpu()
                else:
                    temp_results[key] = results[key][i].detach().cpu()
                    # attention need to be implemented
            self.backbone_frozen_output_cache[descritption] = temp_results

    def load_backbone_output_cache(self, data):
        name_list = data["description"]
        # load cache
        results = {}
        for key in self.backbone_frozen_output_cache[name_list[0]].keys():
            if isinstance(self.backbone_frozen_output_cache[name_list[0]][key], dict):
                results[key] = {}
                for sub_key in self.backbone_frozen_output_cache[name_list[0]][key].keys():
                    temp_results = []
                    for i, descritption in enumerate(name_list):
                        temp_results.append(self.backbone_frozen_output_cache[descritption][key][sub_key].to(data["token"].float()))
                    try:
                        results[key][sub_key] = torch.stack(temp_results, dim=0)
                    except:
                        results[key][sub_key] = self.stack_variable_length_tensors(temp_results)
            else:
                temp_results = []
                for i, descritption in enumerate(name_list):
                    temp_results.append(self.backbone_frozen_output_cache[descritption][key].to(data["token"].float()))
                try:
                    results[key] = torch.stack(temp_results, dim=0)
                except:
                    results[key] = self.stack_variable_length_tensors(temp_results)
        return results

    def stack_variable_length_tensors(self, vl_tensors):
        dim_value = []
        for t in vl_tensors:
            dim_value.append(t.shape)
        dim_max_value = torch.Tensor(dim_value).max(dim=0)[0].int().numpy().tolist()

        pad_tensors = []
        for t in vl_tensors:
            pad_shape = []
            for d_i, d_max in enumerate(dim_max_value):
                pad_shape.append(d_max-t.shape[d_i])
                pad_shape.append(0)
            pad_shape.reverse()   # order need from left to top
            pad_tensors.append(torch.nn.functional.pad(t, pad_shape, mode="constant"))
        output = torch.stack(pad_tensors, dim=0)
        return output

    # load parameter
    def load_param(self, load_choice, model_path):
        param_dict = torch.load(model_path, map_location="cpu")  #["model"]
        if param_dict.get("model") is not None and param_dict.get("args") is not None:
            print("Does not reload weights from official pre-trained file!")
            return 1

        if load_choice == "backbone":
            base_dict = self.backbone.state_dict()
            for i in param_dict:
                module_name = i.replace("backbone.", "")
                if module_name not in self.backbone.state_dict():
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue
                self.backbone.state_dict()[module_name].copy_(param_dict[i])
            print("Complete Load Weight")

        elif load_choice == "overall":
            map_dict = {}
            overall_dict = self.state_dict()
            for key in overall_dict.keys():
                if ":" in key:
                    sub_keys = key.split(":")
                    module_master_name = sub_keys[0]
                    module_branch_name = sub_keys[1].split(".", 1)[1]
                    module_name = module_master_name + "." + module_branch_name
                else:
                    module_name = key
                map_dict[module_name] = {"tar": key}

            for key in param_dict.keys():
                if ":" in key:
                    sub_keys = key.split(":")
                    module_master_name = sub_keys[0]
                    module_branch_name = sub_keys[1].split(".", 1)[1]
                    module_name = module_master_name + "." + module_branch_name
                else:
                    module_name = key
                try:
                    map_dict[module_name]["src"] = key
                except:
                    print("Cannot load %s, Maybe you are using incorrect framework" % key)

            for i in map_dict.keys():
                self.state_dict()[map_dict[i]["tar"]].copy_(param_dict[map_dict[i]["src"]])

            """
            for i in param_dict:
                if i in self.state_dict():
                    self.state_dict()[i].copy_(param_dict[i])
                elif "base."+i in self.state_dict():
                    self.state_dict()["base."+i].copy_(param_dict[i])
                elif "backbone."+i in self.state_dict():
                    self.state_dict()["backbone."+i].copy_(param_dict[i])
                elif i.replace("base", "backbone") in self.state_dict():
                    self.state_dict()[i.replace("base", "backbone")].copy_(param_dict[i])
                else:
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue
            """
            print("Complete Load Weight")

        elif load_choice == "none":
            print("Do not reload Weight by myself.")


    def count_param(model, input_shape=(3, 224, 224)):
        with torch.cuda.device(0):
            flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            return ('{:<30}  {:<8}'.format('Computational complexity: ', flops)) + (
                '{:<30}  {:<8}'.format('Number of parameters: ', params))