# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import numpy as np

def record_item(summary_writer, tag, item, step):
    if isinstance(item, int) or isinstance(item, float):
        # scalar
        summary_writer.add_scalar(tag, item, step)
    elif isinstance(item, dict):
        # scalars
        # summary_writer.add_scalars(tag, item, step)  # the plots are too disordered when the keys are too many
        for key in item.keys():
            if isinstance(item[key], int) or isinstance(item[key], float):
                summary_writer.add_scalar(tag+"-"+key, item[key], step)
    elif isinstance(item, np.ndarray):
        # numpy - images
        if len(item.shape) == 3:
            summary_writer.add_image(tag, item, step, dataformats='HWC')
    else:
        raise Exception("Unknown Data Type to be recorded in Tensorboard Summary File!")

    summary_writer.flush()


def record_dict_into_tensorboard(summary_writer, record_dict, step):
    """
    :param summary_writer:
    :param dict: tensorboard只支持最多两个层级, 当record多于两个层级时可以将多余的key进行串联
    :return:
    """

    for key in record_dict.keys():
        if isinstance(record_dict[key], dict):
            for sub_name in record_dict[key].keys():
                tag = key + "/" + sub_name
                item = record_dict[key][sub_name]
                record_item(summary_writer, tag, item, step)
        else:
            tag = key
            item = record_dict[key]
            record_item(summary_writer, tag, item, step)

    summary_writer.flush()








