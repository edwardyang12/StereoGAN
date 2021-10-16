"""
Author: Isabella Liu 7/18/21
Feature: Config file for messy-table-dataset
"""

from yacs.config import CfgNode as CN

_C = CN()
cfg = _C

# Directories
_C.DIR = CN()
_C.DIR.DATASET = '/code/dataset_local_v9/training'  #  directory of your training dataset

# Split files
_C.SPLIT = CN()
_C.SPLIT.TRAIN = '/code/dataset_local_v9/training_lists/all_train.txt'  # training lists of your training dataset
_C.SPLIT.VAL = '/code/dataset_local_v9/training_lists/all_val.txt'  # training lists of your validation dataset
_C.SPLIT.OBJ_NUM = 18  # Note: table + ground - 17th

_C.SPLIT.LEFT = '0128_irL_kuafuv2_half.png'
_C.SPLIT.LEFT_NO_IR = '0128_irL_kuafu_half_no_ir.png'
_C.SPLIT.RIGHT = '0128_irR_kuafuv2_half.png'
_C.SPLIT.RIGHT_NO_IR = '0128_irR_kuafu_half_no_ir.png'
_C.SPLIT.DEPTH = 'depth.png'
_C.SPLIT.DEPTHL = 'depthL.png'
_C.SPLIT.DEPTHR = 'depthR.png'
_C.SPLIT.META = 'meta.pkl'
_C.SPLIT.LABEL = 'irL_label_image.png'

_C.SPLIT.SIM_REALSENSE = '0128_depth_denoised.png'
_C.SPLIT.REAL_REALSENSE = '1024_depth_real.png'

# Configuration for testing on real dataset
_C.REAL = CN()
_C.REAL.DATASET = '/code/real_dataset_local_v9'  # path to your real testing dataset
_C.REAL.LEFT = '1024_irL_real_1080.png'
_C.REAL.RIGHT = '1024_irR_real_1080.png'
_C.REAL.PAD_WIDTH = 960
_C.REAL.PAD_HEIGHT = 544

# Solver args
_C.SOLVER = CN()
_C.SOLVER.LR_CASCADE = 0.001            # base learning rate for cascade
_C.SOLVER.LR_G = 0.0002                 # base learning rate for G
_C.SOLVER.LR_D = 0.0003                 # base learning rate for D
_C.SOLVER.LR_EPOCHS = '5,10,15:2'       # the epochs to decay lr: the downscale rate
_C.SOLVER.LR_GAN_STEPS = '5,10,15:2'    # the steps to decay lr: the downscale rate
_C.SOLVER.LR_STEPS = '5,10,15:2'        # the steps to decay lr: the downscale rate
_C.SOLVER.EPOCHS = 20                   # number of epochs to train
_C.SOLVER.STEPS = 10000                 # number of steps to train
_C.SOLVER.BATCH_SIZE = 1                # batch size
_C.SOLVER.NUM_WORKER = 1                # num_worker in dataloader

# Model args
_C.ARGS = CN()
_C.ARGS.MAX_DISP = 192              # maximum disparity
_C.ARGS.MODEL = 'gwcnet-c'
_C.ARGS.GRAD_METHOD = 'detach'
_C.ARGS.NDISP = (48, 24)            # ndisps
_C.ARGS.DISP_INTER_R = (4, 1)       # disp_intervals_ratio
_C.ARGS.DLOSSW = (0.5, 2.0)         # depth loss weight for different stage
_C.ARGS.CR_BASE_CHS = (32, 32, 16)  # cost regularization base channels
_C.ARGS.USING_NS = True             # using neighbor search
_C.ARGS.NS_SIZE = 3                 # nb_size
_C.ARGS.CROP_HEIGHT = 256           # crop height
_C.ARGS.CROP_WIDTH = 512            # crop width

# Data Augmentation
_C.DATA_AUG = CN()
_C.DATA_AUG.BRIGHT_MIN = 0.4
_C.DATA_AUG.BRIGHT_MAX = 1.4
_C.DATA_AUG.CONTRAST_MIN = 0.8
_C.DATA_AUG.CONTRAST_MAX = 1.2
_C.DATA_AUG.GAUSSIAN_MIN = 0.1
_C.DATA_AUG.GAUSSIAN_MAX = 2.0
_C.DATA_AUG.GAUSSIAN_KERNEL = 9
