# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com

The required parameters include model_path, data_path, save_path, save_type

Given the input sequences, output and save specific predictions

"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn
import numpy as np
import random

sys.path.append('.')
sys.path.append('..')

from data import make_data_loader

from model import build_model
from engine.predictor import do_prediction

from config import cfg
from utils.logger import setup_logger


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def predict(cfg, save_embeddings=False, save_embeddings_format="raw", save_frequency=1, save_file_prefix="",
            threshold=0.5, allow_nc=True, allow_vis=True):
    # build model and load parameter
    model = build_model(cfg)
    if cfg.EVAL.WEIGHT_PATH != "none":
        model.load_param("overall", cfg.EVAL.WEIGHT_PATH)

    # prepare dataset
    input_data_loader, _, _ = make_data_loader(cfg, model.backbone_alphabet, is_train=False)

    # build and launch engine for evaluation
    Eval_Record = do_prediction(cfg,
                               model,
                               input_data_loader,
                               save_embeddings=save_embeddings,
                               save_embeddings_format=save_embeddings_format,
                               save_results=True,
                               save_frequency=save_frequency,
                               save_file_prefix=save_file_prefix,
                               threshold=threshold,
                               allow_noncanonical_pairs=allow_nc,
                               allow_visualization=allow_vis
                               )

    # logging with tensorboard summaryWriter
    #model_epoch = cfg.EVAL.WEIGHT_PATH.split('/')[-1].split('.')[0].split('_')[-1]
    #model_iteration = len(input_data_loader) * int(model_epoch) if model_epoch.isdigit() == True else 0

    #writer_test = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/predict")
    #record_dict_into_tensorboard(writer_test, Eval_Record, model_iteration)
    #writer_test.close()

    # record in xlsx
    """
    df = pd.DataFrame([value], columns=col_names)
    xls_filename = os.path.join(cfg.SOLVER.OUTPUT_DIR, "{}.xlsx".format(csv_name))
    if os.path.exists(xls_filename) != True:
        with pd.ExcelWriter(xls_filename, engine="openpyxl", mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(xls_filename, engine="openpyxl", mode='a') as writer:
            wb = writer.book
            if sheet_name in wb.sheetnames:
                old_df = pd.read_excel(xls_filename, sheet_name=sheet_name, index_col=0)
                # remove old sheet, otherwise generate new sheets with suffix "1", "2",...
                wb.remove(wb[sheet_name])
                df = pd.concat([old_df, df], axis=0, ignore_index=True)
                df.to_excel(writer, sheet_name=sheet_name)
            else:
                df.to_excel(writer, sheet_name=sheet_name)
    #"""


def main():
    parser = argparse.ArgumentParser(description="Classification Baseline Inference")
    parser.add_argument(
        "--config_file", default=None, help="path to config file", type=str
    )
    parser.add_argument(
        "--data_path", default=None, help="path to data file or folder", type=str
    )
    parser.add_argument(
        "--save_dir", default=None, help="path to savings", type=str
    )
    parser.add_argument(
        "--model_file", default=None, help="file path to model", type=str
    )
    parser.add_argument(
        "--save_frequency", default=1, help="file path to model", type=int
    )
    parser.add_argument(
        "--save_embeddings", action='store_true'
    )
    parser.add_argument(
        "--save_embeddings_format", default="raw", choices=["raw", "mean"]
    )
    parser.add_argument(
        "--save_file_prefix", default="", type=str
    )
    parser.add_argument(
        "--threshold", default=-1, type=float,
    )
    parser.add_argument(
        "--forbid_nc", action='store_true'
    )
    parser.add_argument(
        "--allow_vis", action='store_true'
    )
    parser.add_argument(
        "--device", default="gpu", choices=["cpu", "gpu"]
    )
    parser.add_argument(
        "--gpu_id", default=0, type=int,
    )


    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # custom defined symbols parsing
    # (1). eval batch size
    if cfg.EVAL.DATALOADER.BATCH_SIZE < 0:
        cfg.EVAL.DATALOADER.BATCH_SIZE = -1 * cfg.EVAL.DATALOADER.BATCH_SIZE * cfg.DATA.DATALOADER.BATCH_SIZE

    # (2). parser config name as work space
    if args.config_file != "":
        config_info = args.config_file.strip(".yml").split("/")
        config_parse_dir = ""
        for c_i in config_info:
            if c_i == "CONFIGs":
                config_parse_dir += "/"
            elif config_parse_dir != "":
                config_parse_dir = config_parse_dir + c_i + "/"
            else:
                continue
        cfg.SOLVER.OUTPUT_DIR = cfg.SOLVER.OUTPUT_DIR.replace("${CONFIG_NAME}", config_parse_dir.strip("/"))
        weight_state = "randinit" if cfg.MODEL.BACKBONE_RANDOM_INITIALIZATION == 1 else "pretrain"
        cfg.SOLVER.OUTPUT_DIR = cfg.SOLVER.OUTPUT_DIR.replace("${WEIGHT_STATE}", weight_state)
        stem_freeze = "featbase" if cfg.MODEL.BACKBONE_FROZEN == 1 else "finetune"
        cfg.SOLVER.OUTPUT_DIR = cfg.SOLVER.OUTPUT_DIR.replace("${STEM_FREEZE}", stem_freeze)

    # (3). eval weight path
    cfg.EVAL.WEIGHT_PATH = cfg.EVAL.WEIGHT_PATH.replace("${OUTPUT_DIR}", cfg.SOLVER.OUTPUT_DIR)

    # predict specific setting
    cfg.DATA.DATASETS.NAMES = "custom_seq_L:[1, 1022]_D:[-1,-1]"
    if args.data_path is not None:
        cfg.DATA.DATASETS.ROOT_DIR = args.data_path
    else:
        raise Exception("Need Specify Data Path (file or folder)")
    if args.save_dir is not None:
        cfg.SOLVER.OUTPUT_DIR = args.save_dir
    else:
        #raise Exception("Need Specify Save Dir")
        pass
    if args.model_file is not None:
        cfg.EVAL.WEIGHT_PATH = args.model_file

    if args.threshold != -1:
        threshold = args.threshold
        cfg.MODEL.THRESHOLD = threshold
    else:
        threshold = cfg.MODEL.THRESHOLD

    if args.device == "cpu":
        cfg.MODEL.DEVICE = "cpu"
    else:
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.DEVICE_ID = (args.gpu_id,)

    cfg.freeze()

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("prediction", output_dir, "prediction", 0)
    if args.device == "gpu":
        logger.info("Using {} GPUS, GPU ID: {}".format(num_gpus, args.gpu_id))
    else:
        logger.info("Using CPU")
    #logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))

    logger.info("Model File:{}".format(cfg.EVAL.WEIGHT_PATH))
    logger.info("Batch Size:{}".format(cfg.EVAL.DATALOADER.BATCH_SIZE))
    logger.info("Threshold:{}".format(threshold))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join("%s"%i for i in cfg.MODEL.DEVICE_ID)   # int tuple -> str # cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    logger.info("Prediction Dataset: {}".format(cfg.DATA.DATASETS.ROOT_DIR))
    allow_nc = not args.forbid_nc
    predict(cfg, args.save_embeddings, args.save_embeddings_format, args.save_frequency, args.save_file_prefix,
            threshold, allow_nc, args.allow_vis)


if __name__ == '__main__':
    seed_torch(2018)
    main()
