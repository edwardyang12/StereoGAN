"""
Author: Isabella Liu 8/14/21
Feature: Train cascade with pre-trained cycle GAN
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
from nets.cycle_gan import CycleGANModel
from nets.cascadenet import CascadeNet, cascade_loss
from utils.cascade_metrics import compute_err_metric
from utils.warp_ops import apply_disparity_cu
from utils.config import cfg
from utils.reduce import set_random_seed, synchronize, AverageMeterDict, \
    tensor2float, tensor2numpy, reduce_scalar_outputs, make_nograd_func
from utils.util import setup_logger, weights_init, \
    adjust_learning_rate, save_scalars, save_scalars_graph, save_images, save_images_grid, disp_error_img

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Simple GAN with Cascade Stereo Network (CasStereoNet)')
parser.add_argument('--config-file', type=str, default='./configs/local_train.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--gan-model', type=str, default='', required=True,
                    metavar='FILE', help='Path to pretrained GAN model')
parser.add_argument('--update-g-freq', type=int, default=160, help='Frequency of updating discriminator')
parser.add_argument('--summary-freq', type=int, default=640, help='Frequency of saving temporary results')
parser.add_argument('--save-freq', type=int, default=1, help='Frequency of saving checkpoint')
parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument("--local_rank", type=int, default=0, help='Rank of device in distributed training')
parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')
parser.add_argument('--warp-op', action='store_true',default=True, help='whether use warp_op function to get disparity')
parser.add_argument("--cascade-step", type=int, default=2000, help='Steps to trian GAN alone')

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

# python -m torch.distributed.launch train_cascade_with_pretrained_cycleGAN.py --config-file configs/remote_train.yaml --summary-freq 32 --logdir ../train_8_17_cascade/debug --debug --gan-model


def train(gan_model, cascade_model, cascade_optimizer, TrainImgLoader, ValImgLoader):
    cur_err = np.inf    # store best result
    gan_model.eval()    # set gan model to eval mode

    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # Adjust learning rate
        adjust_learning_rate(cascade_optimizer, epoch_idx, cfg.SOLVER.LR_CASCADE, cfg.SOLVER.LR_EPOCHS)

        # One epoch training loop
        avg_train_scalars_gan = AverageMeterDict()
        avg_train_scalars_cascade = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (len(TrainImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE
            do_summary = global_step % args.summary_freq == 0
            # Train one sample
            scalar_outputs_gan, scalar_outputs_cascade, img_outputs_gan, img_outputs_cascade = \
                train_sample(sample, gan_model, cascade_model, cascade_optimizer, isTrain=True)
            # Save result to tensorboard
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs_gan = tensor2float(scalar_outputs_gan)
                scalar_outputs_cascade = tensor2float(scalar_outputs_cascade)
                avg_train_scalars_gan.update(scalar_outputs_gan)
                avg_train_scalars_cascade.update(scalar_outputs_cascade)
                if do_summary:
                    # Update GAN images
                    save_images_grid(summary_writer, 'train_gan', img_outputs_gan, global_step)
                    # Update GAN losses
                    save_scalars_graph(summary_writer, 'train_gan', scalar_outputs_gan, global_step)
                    # Update Cascade images
                    save_images(summary_writer, 'train_cascade', img_outputs_cascade, global_step)
                    # Update Cascade losses
                    scalar_outputs_cascade.update({'lr': cascade_optimizer.param_groups[0]['lr']})
                    save_scalars(summary_writer, 'train_cascade', scalar_outputs_cascade, global_step)

        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metric_gan = avg_train_scalars_gan.mean()
            total_err_metric_cascade = avg_train_scalars_cascade.mean()
            logger.info(f'Epoch {epoch_idx} train gan    : {total_err_metric_gan}')
            logger.info(f'Epoch {epoch_idx} train cascade: {total_err_metric_cascade}')

            # Save checkpoints
            if (epoch_idx + 1) % args.save_freq == 0:
                checkpoint_data = {
                    'epoch': epoch_idx,
                    'G_A': gan_model.netG_A.state_dict(),
                    'G_B': gan_model.netG_B.state_dict(),
                    'D_A': gan_model.netD_A.state_dict(),
                    'D_B': gan_model.netD_B.state_dict(),
                    'Cascade': cascade_model.state_dict(),
                    'optimizerG': gan_model.optimizer_G.state_dict(),
                    'optimizerD': gan_model.optimizer_D.state_dict(),
                    'optimizerCascade': cascade_optimizer.state_dict()
                }
                save_filename = os.path.join(args.logdir, 'models', f'model_{epoch_idx}.pth')
                torch.save(checkpoint_data, save_filename)
        gc.collect()

        # One epoch validation loop
        avg_val_scalars_gan = AverageMeterDict()
        avg_val_scalars_cascade = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            global_step = (len(ValImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE
            do_summary = global_step % args.summary_freq == 0
            scalar_outputs_gan, scalar_outputs_cascade, img_outputs_gan, img_outputs_cascade = \
                train_sample(sample, gan_model, cascade_model, cascade_optimizer, isTrain=False)
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs_gan = tensor2float(scalar_outputs_gan)
                scalar_outputs_cascade = tensor2float(scalar_outputs_cascade)
                avg_val_scalars_gan.update(scalar_outputs_gan)
                avg_val_scalars_cascade.update(scalar_outputs_cascade)
                if do_summary:
                    save_images_grid(summary_writer, 'val_gan', img_outputs_gan, global_step)
                    scalar_outputs_gan.update({'lr_G': gan_model.optimizer_G.param_groups[0]['lr']})
                    scalar_outputs_gan.update({'lr_D': gan_model.optimizer_D.param_groups[0]['lr']})
                    save_scalars_graph(summary_writer, 'val_gan', scalar_outputs_gan, global_step)
                    save_images(summary_writer, 'val_cascade', img_outputs_cascade, global_step)
                    scalar_outputs_cascade.update({'lr': cascade_optimizer.param_groups[0]['lr']})
                    save_scalars(summary_writer, 'val_cascade', scalar_outputs_cascade, global_step)

        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metric_gan = avg_val_scalars_gan.mean()
            total_err_metric_cascade = avg_val_scalars_cascade.mean()
            logger.info(f'Epoch {epoch_idx} val   gan    : {total_err_metric_gan}')
            logger.info(f'Epoch {epoch_idx} val   cascade: {total_err_metric_cascade}')

            # Save best checkpoints
            new_err = total_err_metric_cascade['depth_abs_err'][0]
            if new_err < cur_err:
                cur_err = new_err
                checkpoint_data = {
                    'epoch': epoch_idx,
                    'G_A': gan_model.netG_A.state_dict(),
                    'G_B': gan_model.netG_B.state_dict(),
                    'D_A': gan_model.netD_A.state_dict(),
                    'D_B': gan_model.netD_B.state_dict(),
                    'Cascade': cascade_model.state_dict(),
                    'optimizerG': gan_model.optimizer_G.state_dict(),
                    'optimizerD': gan_model.optimizer_D.state_dict(),
                    'optimizerCascade': cascade_optimizer.state_dict()
                }
                save_filename = os.path.join(args.logdir, 'models', f'model_best.pth')
                torch.save(checkpoint_data, save_filename)
        gc.collect()


def train_sample(sample, gan_model, cascade_model, cascade_optimizer, isTrain=True):
    if isTrain:
        cascade_model.train()
    else:
        cascade_model.eval()

    # Compute img_L, img_R using GAN
    img_L = sample['img_L'].to(cuda_device)  # [bs, 1, H, W]
    img_R = sample['img_R'].to(cuda_device)  # [bs, 1, H, W]
    img_real = sample['img_real'].to(cuda_device)  # [bs, 1, 2H, 2W]
    img_real = F.interpolate(img_real, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)
    input_sample = {'img_L': img_L, 'img_R': img_R, 'img_real': img_real}
    gan_model.set_input(input_sample)
    with torch.no_grad():
        gan_model.forward()

    # Train on Cascade
    img_L = gan_model.fake_B_L.to(cuda_device)    # [bs, 1, H, W]
    img_R = gan_model.fake_B_R.to(cuda_device)    # [bs, 1, H, W]
    disp_gt = sample['img_disp_l'].to(cuda_device)
    depth_gt = sample['img_depth_l'].to(cuda_device)  # [bs, 1, H, W]
    img_focal_length = sample['focal_length'].to(cuda_device)
    img_baseline = sample['baseline'].to(cuda_device)
    if args.warp_op:
        img_disp_r = sample['img_disp_r'].to(cuda_device)
        disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        del img_disp_r
    # Resize the 2x resolution disp and depth back to H * W
    # Note: This step should go after the apply_disparity_cu
    disp_gt = F.interpolate(disp_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False).squeeze(1)  # [bs, H, W]
    depth_gt = F.interpolate(depth_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]

    mask = (disp_gt < cfg.ARGS.MAX_DISP) * (disp_gt > 0)  # Note in training we do not exclude bg
    if isTrain:
        outputs = cascade_model(img_L, img_R)
        loss_cascade = cascade_loss(outputs, disp_gt, mask, dlossw=[float(e) for e in cfg.ARGS.DLOSSW])
    else:
        with torch.no_grad():
            outputs = cascade_model(img_L, img_R)
            loss_cascade = torch.tensor(0, dtype=img_L.dtype, device=cuda_device, requires_grad=False)
    outputs_stage = outputs["stage{}".format(num_stage)]
    disp_pred = outputs_stage['pred']  # [bs, H, W]
    del outputs

    # Backward and optimization
    if isTrain:
        cascade_optimizer.zero_grad()           # set cascade gradient to zero
        loss_cascade.backward()                   # calculate gradient
        cascade_optimizer.step()                # update cascade weights
        # No backward for GAN
        gan_model.compute_loss_G()
        gan_model.compute_loss_D_A()
        gan_model.compute_loss_D_B()

    # Save gan scalar outputs and images
    scalar_outputs_gan = {
        'G_A': gan_model.loss_G_A, 'G_B': gan_model.loss_G_B,
        'cycle_A': gan_model.loss_cycle_A, 'cycle_B': gan_model.loss_cycle_B,
        'idt_A': gan_model.loss_idt_A, 'idt_B': gan_model.loss_idt_B,
        'D_A': gan_model.loss_D_A, 'D_B': gan_model.loss_D_B
    }
    img_outputs_gan = {
        'img_L': {
            'input': gan_model.real_A_L, 'fake': gan_model.fake_B_L, 'rec': gan_model.rec_A_L, 'idt': gan_model.idt_B_L
        },
        'img_R': {
            'input': gan_model.real_A_R, 'fake': gan_model.fake_B_R, 'rec': gan_model.rec_A_R, 'idt': gan_model.idt_B_R
        },
        'img_Real': {
            'input': gan_model.real_B, 'fake': gan_model.fake_A, 'rec': gan_model.rec_B, 'idt': gan_model.idt_A
        }
    }

    # Compute cascade error metrics
    scalar_outputs_cascade = {'loss': loss_cascade.item()}
    err_metrics = compute_err_metric(disp_gt.unsqueeze(1),
                                     depth_gt,
                                     disp_pred.unsqueeze(1),
                                     img_focal_length,
                                     img_baseline,
                                     mask.unsqueeze(1))
    scalar_outputs_cascade.update(err_metrics)
    # Compute error images
    pred_disp_err_np = disp_error_img(disp_pred.unsqueeze(1), disp_gt.unsqueeze(1), mask.unsqueeze(1))
    pred_disp_err_tensor = torch.from_numpy(np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2])))
    img_outputs_cascade = {
        'disp_gt': disp_gt.unsqueeze(1).repeat([1, 3, 1, 1]),
        'disp_pred': disp_pred.unsqueeze(1).repeat([1, 3, 1, 1]),
        'disp_err': pred_disp_err_tensor
    }

    if is_distributed:
        scalar_outputs_gan = reduce_scalar_outputs(scalar_outputs_gan, cuda_device)
        scalar_outputs_cascade = reduce_scalar_outputs(scalar_outputs_cascade, cuda_device)
    return scalar_outputs_gan, scalar_outputs_cascade, img_outputs_gan, img_outputs_cascade


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

    # Create GAN model
    gan_model = CycleGANModel()
    gan_model.set_device(cuda_device)
    gan_model.set_distributed(is_distributed=is_distributed, local_rank=args.local_rank)
    gan_model.load_model(args.gan_model)

    # Create Cascade model
    cascade_model = CascadeNet(
        maxdisp=cfg.ARGS.MAX_DISP,
        ndisps=[int(nd) for nd in cfg.ARGS.NDISP],
        disp_interval_pixel=[float(d_i) for d_i in cfg.ARGS.DISP_INTER_R],
        cr_base_chs=[int(ch) for ch in cfg.ARGS.CR_BASE_CHS],
        grad_method=cfg.ARGS.GRAD_METHOD,
        using_ns=cfg.ARGS.USING_NS,
        ns_size=cfg.ARGS.NS_SIZE
    ).to(cuda_device)
    cascade_optimizer = torch.optim.Adam(cascade_model.parameters(), lr=cfg.SOLVER.LR_CASCADE, betas=(0.9, 0.999))
    if is_distributed:
        cascade_model = torch.nn.parallel.DistributedDataParallel(
            cascade_model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        cascade_model = torch.nn.DataParallel(cascade_model)

    # Start training
    train(gan_model, cascade_model, cascade_optimizer, TrainImgLoader, ValImgLoader)
