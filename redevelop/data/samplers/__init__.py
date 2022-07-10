# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler

def build_sampler(sampler_name, data_source, is_train=True, distributed_world_size=1, **kwargs):
    """
    :param sampler_name: the name of sampler
    :param data_source: data source
    :param is_train: during training, we allow different samplers; otherwise, we only choose sequential
    :param kwargs: "num_categories_per_batch", "num_instances_per_category" for ClassBalanceSampler
    :return:
    """
    #if len(data_source) == 0:
    #    return None

    # CJY at 2021.5.14 for DDP
    if distributed_world_size > 1:
        sampler = DistributedSampler(data_source)
        return sampler

    if is_train == True:
        if sampler_name == "sequential":
            sampler = SequentialSampler(data_source)
        elif sampler_name == "random":
            sampler = RandomSampler(data_source, replacement=False)
        else:
            raise Exception("Wrong Sampler Name!")
    else:
        sampler = SequentialSampler(data_source)

    return sampler