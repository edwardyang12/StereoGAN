"""
Author: Isabella Liu 8/23/21
Feature: Train simple DA on messytable dataset
"""

import gc
import os
import argparse
import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from datasets.messytable import MessytableDataset
from nets.simple_da import SimpleDA
from utils.config import cfg
from utils.reduce import set_random_seed, synchronize, AverageMeterDict, \
    tensor2float, tensor2numpy, reduce_scalar_outputs, make_nograd_func
from utils.util import setup_logger, weights_init, \
    adjust_learning_rate, save_scalars_graph, save_images_grid

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Simple GAN with Cascade Stereo Network (CasStereoNet)')
parser.add_argument('--config-file', type=str, default='./configs/local_train.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--update-g-freq', type=int, default=160, help='Frequency of updating discriminator')
parser.add_argument('--summary-freq', type=int, default=640, help='Frequency of saving temporary results')
parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of saving checkpoint')
parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument("--local_rank", type=int, default=0, help='Rank of device in distributed training')
parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')

args = parser.parse_args()
cfg.merge_from_file(args.config_file)
num_stage = len([int(nd) for nd in cfg.ARGS.NDISP])  # number of stages in cascade network

# Set random seed to make sure networks in different processes are same
set_random_seed(args.seed)

# Set up distributed training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
args.is_distributed = is_distributed
if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
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


# python -m torch.distributed.launch train_simpleDA.py --config-file configs/remote_train_gan.yaml --summary-freq 32 --logdir ../train_8_23_da/debug --debug


def train(model, TrainImgLoader, ValImgLoader):
    cur_err = np.inf  # store best result

    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (len(TrainImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE
            if global_step > cfg.SOLVER.STEPS:
                break

            # Adjust learning rate
            adjust_learning_rate(model.optimizer, global_step, cfg.SOLVER.LR_G, cfg.SOLVER.LR_STEPS)

            do_summary = global_step % args.summary_freq == 0
            scalar_outputs, img_outputs = train_DA_sample(sample, model)
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs = tensor2float(scalar_outputs)
                avg_train_scalars.update(scalar_outputs)
                if do_summary:
                    save_images_grid(summary_writer, 'train', img_outputs, global_step)
                    save_scalars_graph(summary_writer, 'train', scalar_outputs, global_step)
                    summary_writer.add_scalar('train/lr', model.optimizer.param_groups[0]['lr'], global_step)

                # Save checkpoints
                if (global_step + 1) % args.save_freq == 0:
                    checkpoint_data = {
                        'epoch': epoch_idx,
                        'feature_extractor': model.feature_extractor.state_dict(),
                        'D': model.net_D.state_dict(),
                        'optimizer': model.optimizer.state_dict(),
                    }
                    save_filename = os.path.join(args.logdir, 'models', f'model_{global_step}.pth')
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric = avg_train_scalars.mean()
                    avg_train_scalars = AverageMeterDict()
                    logger.info(f'Step {global_step} train total_err_metrics: {total_err_metric}')
                    gc.collect()

        # One epoch validation loop
        avg_val_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            global_step = (len(ValImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE
            do_summary = global_step % args.summary_freq == 0
            scalar_outputs, img_outputs = test_DA_sample(sample, model)
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs = tensor2float(scalar_outputs)
                avg_val_scalars.update(scalar_outputs)
                if do_summary:
                    save_images_grid(summary_writer, 'val', img_outputs, global_step)
                    save_scalars_graph(summary_writer, 'val', scalar_outputs, global_step)

        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metric = avg_val_scalars.mean()
            logger.info(f'Epoch {epoch_idx} val   total_err_metrics: {total_err_metric}')

            # Save best checkpoints
            new_err = total_err_metric['loss_D'] if is_distributed is False else total_err_metric['loss_D'][0]
            if new_err < cur_err:
                cur_err = new_err
                checkpoint_data = {
                    'epoch': epoch_idx,
                    'feature_extractor': model.feature_extractor.state_dict(),
                    'D': model.net_D.state_dict(),
                    'optimizer': model.optimizer.state_dict(),
                }
                save_filename = os.path.join(args.logdir, 'models', f'model_best.pth')
                torch.save(checkpoint_data, save_filename)
        gc.collect()


# Train a sample batch on GAN
def train_DA_sample(sample, model):
    img_L = sample['img_L'].to(cuda_device)  # [bs, 1, H, W]
    img_R = sample['img_R'].to(cuda_device)  # [bs, 1, H, W]
    img_real = sample['img_real'].to(cuda_device)  # [bs, 1, 2H, 2W]
    img_real = F.interpolate(img_real, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)

    # Set input and perform optimization
    input_sample = {'img_L': img_L, 'img_R': img_R, 'img_real': img_real}
    model.set_input(input_sample)
    model.forward()
    model.backward()

    scalar_outputs = {
        'loss_D': model.loss_D
    }
    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs, cuda_device)

    img_outputs = {
        'img_L_input': {
            'input': img_L
        },
        'img_L_features': {
            'feature_0': model.img_L_feature[:, :1, :, :], 'feature_10': model.img_L_feature[:, 10:11, :, :],
            'feature_20': model.img_L_feature[:, 20:21, :, :]
        },
        'img_R_input': {
            'input': img_R
        },
        'img_R_features': {
            'feature_0': model.img_R_feature[:, :1, :, :], 'feature_10': model.img_R_feature[:, 10:11, :, :],
            'feature_20': model.img_R_feature[:, 20:21, :, :]
        },
        'img_Real_input': {
            'input': img_real
        },
        'img_Real_features': {
            'feature_0': model.img_real_feature[:, :1, :, :], 'feature_10': model.img_real_feature[:, 10:11, :, :],
            'feature_20': model.img_real_feature[:, 20:21, :, :]
        }
    }
    return scalar_outputs, img_outputs


# Train a sample batch on GAN
@make_nograd_func
def test_DA_sample(sample, model):
    img_L = sample['img_L'].to(cuda_device)  # [bs, 1, H, W]
    img_R = sample['img_R'].to(cuda_device)  # [bs, 1, H, W]
    img_real = sample['img_real'].to(cuda_device)  # [bs, 1, 2H, 2W]
    img_real = F.interpolate(img_real, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)

    # Set input and perform optimization
    input_sample = {'img_L': img_L, 'img_R': img_R, 'img_real': img_real}
    model.set_input(input_sample)
    model.forward()

    scalar_outputs = {
        'loss_D': model.loss_D
    }
    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs, cuda_device)

    img_outputs = {
        'img_L_input': {
            'input': img_L
        },
        'img_L_features': {
            'feature_0': model.img_L_feature[:, :1, :, :], 'feature_10': model.img_L_feature[:, 10:11, :, :],
            'feature_20': model.img_L_feature[:, 20:21, :, :]
        },
        'img_R_input': {
            'input': img_R
        },
        'img_R_features': {
            'feature_0': model.img_R_feature[:, :1, :, :], 'feature_10': model.img_R_feature[:, 10:11, :, :],
            'feature_20': model.img_R_feature[:, 20:21, :, :]
        },
        'img_Real_input': {
            'input': img_real
        },
        'img_Real_features': {
            'feature_0': model.img_real_feature[:, :1, :, :], 'feature_10': model.img_real_feature[:, 10:11, :, :],
            'feature_20': model.img_real_feature[:, 20:21, :, :]
        }
    }
    return scalar_outputs, img_outputs


if __name__ == '__main__':
    # Obtain dataloader
    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, debug=args.debug, sub=400)
    val_dataset = MessytableDataset(cfg.SPLIT.VAL, debug=args.debug, sub=200)
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

    # Create model
    model = SimpleDA()
    model.set_device(cuda_device)
    model.set_distributed(is_distributed=is_distributed, local_rank=args.local_rank)

    # Start training
    train(model, TrainImgLoader, ValImgLoader)
