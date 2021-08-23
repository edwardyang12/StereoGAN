"""
Author: Isabella Liu 8/8/21
Feature: Train simple GAN with cascade stereo network
"""

import os
import argparse
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as Transforms

from datasets.messytable import MessytableDataset
from nets.cascadenet import CascadeNet
from nets.discriminator import SimpleD
from nets.generator import SimpleG
from utils.config import cfg
from utils.reduce import set_random_seed, synchronize, AverageMeterDict
from utils.util import setup_logger, adjust_learning_rate

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Simple GAN with Cascade Stereo Network (CasStereoNet)')
parser.add_argument('--config-file', type=str, default='./configs/local_train.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--summary_freq', type=int, default=1000, help='Frequency of saving temporary results')
parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument("--local_rank", type=int, default=0, help='Rank of device in distributed training')
parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')

args = parser.parse_args()
cfg.merge_from_file(args.config_file)
num_stage = len([int(nd) for nd in cfg.ARGS.NDISP])     # number of stages in cascade network

# Set random seed to make sure networks in different processes are same
set_random_seed(args.seed)

# Set up distributed training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
args.is_distributed = is_distributed
if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group( backend="nccl", init_method="env://")
    synchronize()
cuda_device = torch.device("cuda:{}".format(args.local_rank))

# Set up tensorboard and logger
os.makedirs(args.logdir, exist_ok=True)
summary_writer = tensorboardX.SummaryWriter(logdir=args.logdir)
logger = setup_logger("Simple GAN cascade stereo", distributed_rank=args.local_rank, save_dir=args.logdir)
logger.info(f'Loaded config file: \'{args.config_file}\'')
logger.info(f'Running with configs:\n{cfg}')
logger.info(f'Running with {num_gpus} GPUs')


def train(cascade_model, simpleG, simpleD, optimizer_cascade, optimizerG, optimizerD, TrainImgLoader, ValImgLoader):
    cur_err = np.inf    # store best result

    # Initialize Simple GAN loss
    gan_loss = nn.BCELoss()

    # Get random crop transformer to crop real image to patch TODO optional patch size
    random_crop = Transforms.RandomCrop((32, 32))

    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # Adjust learning rate
        adjust_learning_rate(optimizer_cascade, epoch_idx, cfg.SOLVER.LR_CASCADE, cfg.SOLVER.LR_EPOCHS)
        adjust_learning_rate(optimizerG, epoch_idx, cfg.SOLVER.LR_GAN, cfg.SOLVER.LR_EPOCHS)
        adjust_learning_rate(optimizerD, epoch_idx, cfg.SOLVER.LR_GAN, cfg.SOLVER.LR_EPOCHS)

        # One epoch training loop
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (len(TrainImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE * num_gpus
            do_summary = global_step % args.summary_freq == 0

            img_L = sample['img_L'].to(cuda_device)         # [bs, 1, H, W]
            img_R = sample['img_R'].to(cuda_device)         # [bs, 1, H, W]
            img_real = sample['img_real'].to(cuda_device)   # [bs, 1, 2H, 2W]
            img_real = F.interpolate(img_real, scale_factor=0.5, mode='bilinear', align_corners=False)

            # Update D
            # Train D with real input, crop real image to patch size 32 * 32
            optimizerD.zero_grad()
            input_real = random_crop(img_real)      # [bs, 1, 32, 32]
            pred_label = simpleD(input_real)
            gt_label = torch.ones_like(pred_label, dtype=torch.float32, device=cuda_device)
            loss_D_real = gan_loss(pred_label, gt_label)
            loss_D_real.backward()
            # Train D with fake input, crop fake image from G to patch size 32 * 32
            input_fake_L = simpleG(img_L)
            input_fake_R = simpleG(img_R)
            input_fake_L = random_crop(input_fake_L)
            input_fake_R = random_crop(input_fake_R)
            pred_label_L = simpleD(input_fake_L.detach())
            pred_label_R = simpleD(input_fake_R.detach())
            gt_label = torch.zeros_like(pred_label_L, dtype=torch.float32, device=cuda_device)
            loss_D_fake_L = gan_loss(pred_label_L, gt_label)
            loss_D_fake_R = gan_loss(pred_label_R, gt_label)
            loss_D_fake = (loss_D_fake_L + loss_D_fake_R) * 0.5
            loss_D_fake.backward()
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            optimizerD.step()

            # Update G
            optimizerG.zero_grad()
            pred_label_L = simpleD(input_fake_L)   # Note no detach here because we are backprop through G
            pred_label_R = simpleD(input_fake_R)
            gt_label = torch.ones_like(pred_label_L, dtype=torch.float32, device=cuda_device)
            loss_G_L = gan_loss(pred_label_L, gt_label)
            loss_G_R = gan_loss(pred_label_R, gt_label)
            loss_G = (loss_G_L + loss_G_R) * 0.5
            loss_G.backward()
            optimizerG.step()







if __name__ == '__main__':
    # Obtain dataloader
    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, debug=args.debug, sub=200)
    val_dataset = MessytableDataset(cfg.SPLIT.VAL, debug=args.debug, sub=100)
    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                                          rank=dist.get_rank())

        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, cfg.SOLVER.BATCH_SIZE, sampler=train_sampler,
                                                     num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True, pin_memory=True)
        ValImgLoader = torch.utils.data.DataLoader(val_dataset, cfg.SOLVER.BATCH_SIZE, sampler=val_sampler,
                                                   num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False, pin_memory=True)
    else:
        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                     shuffle=True, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True)

        ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                   shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)

    # Initialize model
    cascade_model = CascadeNet(
        maxdisp=cfg.ARGS.MAX_DISP,
        ndisps=[int(nd) for nd in cfg.ARGS.NDISP],
        disp_interval_pixel=[float(d_i) for d_i in cfg.ARGS.DISP_INTER_R],
        cr_base_chs=[int(ch) for ch in cfg.ARGS.CR_BASE_CHS],
        grad_method=cfg.ARGS.GRAD_METHOD,
        using_ns=cfg.ARGS.USING_NS,
        ns_size=cfg.ARGS.NS_SIZE
    )
    simpleG = SimpleG()
    simpleD = SimpleD()

    # Set up optimizer
    optimizer_cascade = torch.optim.Adam(cascade_model.parameters(), lr=cfg.SOLVER.LR_CASCADE, betas=(0.9, 0.999))
    optimizerG = torch.optim.Adam(simpleG.parameters(), lr=cfg.SOLVER.LR_GAN, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(simpleD.parameters(), lr=cfg.SOLVER.LR_GAN, betas=(0.5, 0.999))

    # Set cuda device
    for model in [cascade_model, simpleG, simpleD]:
        model = model.cuda()

    # Enable Multiprocess training
    for model in [cascade_model, simpleG, simpleD]:
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank
            )
        else:
            model = torch.nn.DataParallel(model)

    # Start training
    train(cascade_model, simpleG, simpleD, optimizer_cascade, optimizerG, optimizerD, TrainImgLoader, ValImgLoader)

