"""
Author: Isabella Liu 8/8/21
Feature: Train simple GAN on messytable dataset
"""
import gc
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
from nets.discriminator import SimpleD32, SimpleD64
from nets.generator import SimpleG, DownUpG, SimpleUnetG
from nets.temp_discriminator import Discriminator, NLayerDiscriminator
from nets.temp_generator import Generator, ResnetGenerator
from utils.config import cfg
from utils.reduce import set_random_seed, synchronize, AverageMeterDict, \
    tensor2float, tensor2numpy, reduce_scalar_outputs, make_nograd_func
from utils.util import setup_logger, weights_init, \
    adjust_learning_rate, save_images, save_scalars

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Simple GAN with Cascade Stereo Network (CasStereoNet)')
parser.add_argument('--config-file', type=str, default='./configs/local_train.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--update-g-freq', type=int, default=160, help='Frequency of updating discriminator')
parser.add_argument('--summary-freq', type=int, default=640, help='Frequency of saving temporary results')
parser.add_argument('--save-freq', type=int, default=1, help='Frequency of saving checkpoint')
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
os.makedirs(os.path.join(args.logdir, 'models'), exist_ok=True)
summary_writer = tensorboardX.SummaryWriter(logdir=args.logdir)
logger = setup_logger("Simple GAN cascade stereo", distributed_rank=args.local_rank, save_dir=args.logdir)
logger.info(f'Loaded config file: \'{args.config_file}\'')
logger.info(f'Running with configs:\n{cfg}')
logger.info(f'Running with {num_gpus} GPUs')

# python -m torch.distributed.launch train_simpleGAN.py --config-file configs/remote_train.yaml --summary-freq 32 --update-g-freq 10 --logdir ../train_8_11/debug4_patch_d_temp_g --debug


def train(simpleG, simpleD, optimizerG, optimizerD, TrainImgLoader, ValImgLoader):
    cur_err = np.inf    # store best result

    # Initialize Simple GAN loss
    gan_loss = nn.MSELoss()

    # Get random crop transformer to crop real image to patch TODO optional patch size
    random_crop = Transforms.RandomCrop((64, 64))

    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # Adjust learning rate
        adjust_learning_rate(optimizerG, epoch_idx, cfg.SOLVER.LR_G, cfg.SOLVER.LR_EPOCHS)
        adjust_learning_rate(optimizerD, epoch_idx, cfg.SOLVER.LR_D, cfg.SOLVER.LR_EPOCHS)

        # One epoch training loop
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (len(TrainImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE
            do_summary = global_step % args.summary_freq == 0
            update_g = global_step % args.update_g_freq == 0
            scalar_outputs, img_outputs = train_GAN_sample(sample, random_crop, gan_loss,
                                                           simpleG, simpleD, optimizerG, optimizerD, update_g)
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs = tensor2float(scalar_outputs)
                avg_train_scalars.update(scalar_outputs)
                if do_summary:
                    save_images(summary_writer, 'train', img_outputs, global_step)
                    scalar_outputs.update({'lr': optimizerD.param_groups[0]['lr']})
                    save_scalars(summary_writer, 'train', scalar_outputs, global_step)

        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metric = avg_train_scalars.mean()
            logger.info(f'Epoch {epoch_idx} train total_err_metrics: {total_err_metric}')

            # Save checkpoints
            if (epoch_idx + 1) % args.save_freq == 0:
                checkpoint_data = {
                    'epoch': epoch_idx,
                    'G': simpleG.state_dict(),
                    'D': simpleD.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                }
                save_filename = os.path.join(args.logdir, 'models', f'model_{epoch_idx}.pth')
                torch.save(checkpoint_data, save_filename)
        gc.collect()

        # One epoch validation loop
        avg_val_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            global_step = (len(ValImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE
            do_summary = global_step % args.summary_freq == 0
            scalar_outputs, img_outputs = test_GAN_sample(sample, random_crop, gan_loss,
                                                          simpleG, simpleD, optimizerG, optimizerD)
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs = tensor2float(scalar_outputs)
                avg_val_scalars.update(scalar_outputs)
                if do_summary:
                    save_images(summary_writer, 'val', img_outputs, global_step)
                    scalar_outputs.update({'lr': optimizerD.param_groups[0]['lr']})
                    save_scalars(summary_writer, 'val', scalar_outputs, global_step)

        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metric = avg_val_scalars.mean()
            logger.info(f'Epoch {epoch_idx} val   total_err_metrics: {total_err_metric}')

            # Save best checkpoints
            new_err = total_err_metric['loss_G'][0]
            if new_err < cur_err:
                cur_err = new_err
                checkpoint_data = {
                    'epoch': epoch_idx,
                    'G': simpleG.state_dict(),
                    'D': simpleD.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                }
                save_filename = os.path.join(args.logdir, 'models', f'model_best.pth')
                torch.save(checkpoint_data, save_filename)
        gc.collect()


# Train a sample batch on GAN
def train_GAN_sample(sample, random_crop, gan_loss, simpleG, simpleD, optimizerG, optimizerD, updateG=False):
    img_L = sample['img_L'].to(cuda_device)  # [bs, 1, H, W]
    img_R = sample['img_R'].to(cuda_device)  # [bs, 1, H, W]
    img_real = sample['img_real'].to(cuda_device)  # [bs, 1, 2H, 2W]
    img_real = F.interpolate(img_real, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)

    # Update D
    # Train D with real input, crop real image to patch size 32 * 32
    optimizerD.zero_grad()
    input_real_patch = random_crop(img_real)  # [bs, 1, 32, 32]
    pred_label = simpleD(input_real_patch)
    gt_label = torch.ones_like(pred_label, dtype=torch.float32, device=cuda_device)
    loss_D_real = gan_loss(pred_label, gt_label)
    loss_D_real.backward()
    # Train D with fake input, crop fake image from G to patch size 32 * 32
    input_fake_L = simpleG(img_L)
    input_fake_R = simpleG(img_R)
    input_fake_L_patch = input_fake_L
    input_fake_R_patch = input_fake_R
    # input_fake_L_patch = random_crop(input_fake_L)
    # input_fake_R_patch = random_crop(input_fake_R)
    pred_label_L = simpleD(input_fake_L_patch.detach())
    pred_label_R = simpleD(input_fake_R_patch.detach())
    gt_label = torch.zeros_like(pred_label_L, dtype=torch.float32, device=cuda_device)
    loss_D_fake_L = gan_loss(pred_label_L, gt_label)
    loss_D_fake_R = gan_loss(pred_label_R, gt_label)
    loss_D_fake = (loss_D_fake_L + loss_D_fake_R) * 0.5
    loss_D_fake.backward()
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    optimizerD.step()

    # Update G
    optimizerG.zero_grad()
    pred_label_L = simpleD(input_fake_L_patch)  # Note no detach here because we are backprop through G
    pred_label_R = simpleD(input_fake_R_patch)
    gt_label = torch.ones_like(pred_label_L, dtype=torch.float32, device=cuda_device)
    loss_G_L = gan_loss(pred_label_L, gt_label)
    loss_G_R = gan_loss(pred_label_R, gt_label)
    loss_G = (loss_G_L + loss_G_R) * 0.5
    if updateG:
        loss_G.backward()
        optimizerG.step()

    scalar_outputs = {
        'loss_G': loss_G,
        'loss_D': loss_D
    }
    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs, cuda_device)

    img_outputs = {
        'imgL': img_L, 'imgL_GAN': input_fake_L,
        'imgR': img_R, 'imgR_GAN': input_fake_R,
        'imgReal': img_real
    }
    return scalar_outputs, img_outputs


# Train a sample batch on GAN
@make_nograd_func
def test_GAN_sample(sample, random_crop, gan_loss, simpleG, simpleD, optimizerG, optimizerD):
    img_L = sample['img_L'].to(cuda_device)  # [bs, 1, H, W]
    img_R = sample['img_R'].to(cuda_device)  # [bs, 1, H, W]
    img_real = sample['img_real'].to(cuda_device)  # [bs, 1, 2H, 2W]
    img_real = F.interpolate(img_real, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)

    # Test on D with real input
    input_real_patch = random_crop(img_real)  # [bs, 1, 32, 32]
    pred_label = simpleD(input_real_patch)
    gt_label = torch.ones_like(pred_label, dtype=torch.float32, device=cuda_device)
    loss_D_real = gan_loss(pred_label, gt_label)
    # Test D with fake input, crop fake image from G to patch size 32 * 32
    input_fake_L = simpleG(img_L)
    input_fake_R = simpleG(img_R)
    input_fake_L_patch = input_fake_L
    input_fake_R_patch = input_fake_R
    # input_fake_L_patch = random_crop(input_fake_L)
    # input_fake_R_patch = random_crop(input_fake_R)
    pred_label_L = simpleD(input_fake_L_patch.detach())
    pred_label_R = simpleD(input_fake_R_patch.detach())
    gt_label = torch.zeros_like(pred_label_L, dtype=torch.float32, device=cuda_device)
    loss_D_fake_L = gan_loss(pred_label_L, gt_label)
    loss_D_fake_R = gan_loss(pred_label_R, gt_label)
    loss_D_fake = (loss_D_fake_L + loss_D_fake_R) * 0.5
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    # Test on G
    pred_label_L = simpleD(input_fake_L_patch)  # Note no detach here because we are backprop through G
    pred_label_R = simpleD(input_fake_R_patch)
    gt_label = torch.ones_like(pred_label_L, dtype=torch.float32, device=cuda_device)
    loss_G_L = gan_loss(pred_label_L, gt_label)
    loss_G_R = gan_loss(pred_label_R, gt_label)
    loss_G = (loss_G_L + loss_G_R) * 0.5

    scalar_outputs = {
        'loss_G': loss_G,
        'loss_D': loss_D
    }
    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs, cuda_device)

    img_outputs = {
        'imgL': img_L, 'imgL_GAN': input_fake_L,
        'imgR': img_R, 'imgR_GAN': input_fake_R,
        'imgReal': img_real
    }
    return scalar_outputs, img_outputs


if __name__ == '__main__':
    # Obtain dataloader
    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, debug=args.debug, sub=600)
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

    # simpleG = SimpleG()
    # simpleG = SimpleUnetG()
    # simpleD = SimpleD32()
    simpleD = NLayerDiscriminator(input_nc=1)

    # simpleG = Generator()
    simpleG = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, use_dropout=True, n_blocks=9)
    # simpleD = Discriminator()


    # Set up optimizer
    optimizerG = torch.optim.Adam(simpleG.parameters(), lr=cfg.SOLVER.LR_G, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(simpleD.parameters(), lr=cfg.SOLVER.LR_D, betas=(0.5, 0.999))

    # Set cuda device
    for model in [simpleG, simpleD]:
        model = model.cuda()
        # model.apply(weights_init)     # weight init to mean=0, std=0.02

    # Enable Multiprocess training
    for model in [simpleG, simpleD]:
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank
            )
        else:
            model = torch.nn.DataParallel(model)

    # Start training
    train(simpleG, simpleD, optimizerG, optimizerD, TrainImgLoader, ValImgLoader)

