from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1.Data General Setting. Can be replaced in respective sets
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.DATA = CN()
# -----------------------------------------------------------------------------
# Data.Dataset
# -----------------------------------------------------------------------------
_C.DATA.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATA.DATASETS.NAMES = ('none',)
# Root directory where datasets should be used (and downloaded if not found)
_C.DATA.DATASETS.ROOT_DIR = ('./data',)
# -----------------------------------------------------------------------------
# Data.DataLoader
# -----------------------------------------------------------------------------
_C.DATA.DATALOADER = CN()
# Number of data loading threads
_C.DATA.DATALOADER.NUM_WORKERS = 4
# Sampler for data loading
_C.DATA.DATALOADER.SAMPLER = "random"   # "sequential", "random", "weighted_random", "class_balance_random"
# batch size for training
_C.DATA.DATALOADER.BATCH_SIZE = 1

# -----------------------------------------------------------------------------
# Data.TRANSFORM
# -----------------------------------------------------------------------------
_C.DATA.TRANSFORM = CN()
# Sequence Crop Length
_C.DATA.TRANSFORM.SEQUENCE_CROP_SIZE = 2000

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.Model General Setting. Can be replaced in respective sets  Structure Information
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2-1 Basic Config
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = (0, 1)  #it has bug with “both str and int convert to int, so use tuple instead which seems to turn into remove brackets automatically”
# local rank for DDP
_C.MODEL.LOCAL_RANK = -1
# Name of backbone
_C.MODEL.BACKBONE_NAME = 'densenet121'
# Three kinds of Predictors for Protein, DNA and RNA.
_C.MODEL.SEQWISE_PREDICTOR_NAME = "none"

_C.MODEL.ELEWISE_PREDICTOR_NAME = "none"

_C.MODEL.PAIRWISE_PREDICTOR_NAME = "none"

# Regarding weight loading, we initialize for most partitions and load pre-trained model selectively to cover.
# Random initialized for Backbone (downstream modules must be initialized, so we care the backbone)
_C.MODEL.BACKBONE_RANDOM_INITIALIZATION = 0

# Pretrained or not for random initialized model. (Here, the pre-trained means custom pre-training)
_C.MODEL.PRETRAINED = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAINED_MODEL_PATH = r'PATH\unknown.pth'
# Pretrained Type
_C.MODEL.PRETRAINED_MODULE_NAME = "backbone"

# some specific setting for model
# setting bias-free for model # actually implemented in optimizer (zero initialization without subsequent optimization）
_C.MODEL.BIAS_FREE = 0
# freeze backbone
_C.MODEL.BACKBONE_FROZEN = 0

_C.MODEL.THRESHOLD = 0.5

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3.LOSS General Setting. Can be replaced in respective sets  Structure Information
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# LOSS configuration
_C.LOSS = CN()
# The loss type of metric loss
_C.LOSS.TYPE = 'cross_entropy_loss'
_C.LOSS.WEIGHT_MODE = "none"

# METRIC configuration
_C.METRIC = CN()
_C.METRIC.TYPE = 'top-metrics_contact'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4.Solver
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# Path to checkpoint and saved log of trained model
_C.SOLVER.OUTPUT_DIR = "work_space"
# -----------------------------------------------------------------------------
# OPTIMIZER configuration
# -----------------------------------------------------------------------------
_C.SOLVER.OPTIMIZER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER.NAME = "Adam"
# Momentum
_C.SOLVER.OPTIMIZER.MOMENTUM = 0.9
# Settings of weight decay
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0005
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS = 0.
# accumulation_steps (only used in training)
_C.SOLVER.OPTIMIZER.ACCUMULATION_STEP = 1
# -----------------------------------------------------------------------------
# SCHEDULER configuration
# -----------------------------------------------------------------------------
_C.SOLVER.SCHEDULER = CN()
# scheduler step stride or change frequency
_C.SOLVER.SCHEDULER.NAME = "WarmupMultiStepLR"
# the unit of following abstract steps
_C.SOLVER.SCHEDULER.STEP_UNIT = "epoch"           # "epoch", "iteration"
# the frequency of scheduler step forward
_C.SOLVER.SCHEDULER.STEP_FREQUENCY = "iteration"  # "epoch", "iteration"
# Base learning rate - used in optimizer initialization actually
_C.SOLVER.SCHEDULER.BASE_LR = 1e-4
# Factor of learning bias - used in optimizer initialization actually
_C.SOLVER.SCHEDULER.BIAS_LR_FACTOR = 2
# if train from scratch: True or False
_C.SOLVER.SCHEDULER.START_FROM_SCRATCH = 1

# For Warm Up Method
# method of warm up, option: 'constant','linear'
_C.SOLVER.SCHEDULER.WARMUP_METHOD = "linear"
# warm up initial factor
_C.SOLVER.SCHEDULER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.SCHEDULER.WARMUP_STEPS = 500.0

# For Specific Decay Method
# decay rate of learning rate - MultiStepLR
_C.SOLVER.SCHEDULER.GAMMA = 0.1
# decay step of learning rate - MultiStepLR
_C.SOLVER.SCHEDULER.MILESTONES = (30, 55)
# decay step of learning rate - Cosine Triangle
_C.SOLVER.SCHEDULER.MAIN_STEPS = 30

# -----------------------------------------------------------------------------
# APEX configuration
# -----------------------------------------------------------------------------
_C.SOLVER.APEX = CN()
_C.SOLVER.APEX.OPT_LEVEL = "none"   # "O0", "O1", "O2", "O3"

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Eval Configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# -----------------------------------------------------------------------------
# Eval Weight Path
# -----------------------------------------------------------------------------
_C.EVAL.WEIGHT_PATH = ""
# -----------------------------------------------------------------------------
# Eval Dataloader
# -----------------------------------------------------------------------------
_C.EVAL.DATALOADER = CN()
# Number of images per batch
_C.EVAL.DATALOADER.BATCH_SIZE = -1
