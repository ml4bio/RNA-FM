# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from .datasets import build_dataset
from .transforms import build_transforms
from .samplers import build_sampler
from .collate_function import build_collate_fn

from torch.utils.data import DataLoader


def find_dataset_type(dataset_name):
    """
    pre-define the dataset type corresponding to the dataset name, mainly for transformation
    :param dataset_name:
    :return:
    """
    dataset_type = "none"

    return dataset_type


def make_data_loader(cfg, alphabet, is_train):
    if cfg.DATA.DATASETS.NAMES == "none":
        return None, None, None, None

    # 0. config
    dataset_name = cfg.DATA.DATASETS.NAMES
    dataset_type = find_dataset_type(dataset_name)
    root_path = cfg.DATA.DATASETS.ROOT_DIR

    # build transforms
    train_transforms = build_transforms(dataset_type, cfg, is_train=is_train)
    # val_transforms = build_transforms(dataset_type, cfg, is_train=False)
    # test_transforms = build_transforms(dataset_type, cfg, is_train=False)

    # build datasets
    train_set = build_dataset(dataset_name, "train", root_path, train_transforms)
    # val_set = build_dataset(dataset_name, "valid", root_path, val_transforms)
    # test_set = build_dataset(dataset_name, "test", root_path, test_transforms)

    # build samplers
    sampler_name = cfg.DATA.DATALOADER.SAMPLER
    distributed_world_size = len(cfg.MODEL.DEVICE_ID)

    train_sampler = build_sampler(
        sampler_name, train_set, is_train=is_train,
        distributed_world_size=distributed_world_size
    )
    # valid_sampler = build_sampler(
    #     sampler_name, val_set, is_train=False,
    #     distributed_world_size=distributed_world_size
    # )
    # test_sampler = build_sampler(
    #     sampler_name, test_set, is_train=False,
    #     distributed_world_size=distributed_world_size
    # )

    # build collate function
    collate_fn = build_collate_fn(alphabet, train_set.data_type)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    drop_last = False

    train_batch_size = cfg.DATA.DATALOADER.BATCH_SIZE
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    # val_batch_size = cfg.EVAL.DATALOADER.BATCH_SIZE
    # val_loader = DataLoader(
    #     val_set, batch_size=val_batch_size, sampler=valid_sampler,
    #     num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    # )
    #
    # test_batch_size = cfg.EVAL.DATALOADER.BATCH_SIZE
    # test_loader = DataLoader(
    #     test_set, batch_size=test_batch_size, sampler=test_sampler,
    #     num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    # )

    val_loader = None
    test_loader = None

    return train_loader, val_loader, test_loader