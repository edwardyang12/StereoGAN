"""
Author: Isabella Liu 9/7/21
Feature: Train feature reprojection with PSMNet
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
from nets.psmnet import PSMNet
from nets.transformer import Transformer
from utils.cascade_metrics import compute_err_metric
from utils.warp_ops import apply_disparity_cu
from utils.reprojection import get_reprojection_error, apply_disparity
from utils.config import cfg
from utils.reduce import set_random_seed, synchronize, AverageMeterDict, \
    tensor2float, tensor2numpy, reduce_scalar_outputs, make_nograd_func
from utils.util import setup_logger, weights_init, \
    adjust_learning_rate, save_scalars, save_scalars_graph, save_images, save_images_grid, disp_error_img
from nets.FGSM import FastGradientSignUntargeted

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Reprojection with Pyramid Stereo Network (PSMNet)')
parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--summary-freq', type=int, default=500, help='Frequency of saving temporary results')
parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of saving checkpoint')
parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument("--local_rank", type=int, default=0, help='Rank of device in distributed training')
parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')
parser.add_argument('--warp-op', action='store_true',default=True, help='whether use warp_op function to get disparity')
parser.add_argument('--loss-ratio', type=float, default=0.05, help='Ratio between loss_G and loss_cascade')
parser.add_argument('--gaussian-blur', action='store_true',default=False, help='whether apply gaussian blur')
parser.add_argument('--color-jitter', action='store_true',default=False, help='whether apply color jitter')

args = parser.parse_args()
cfg.merge_from_file(args.config_file)

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
logger = setup_logger("Reprojection PSMNet", distributed_rank=args.local_rank, save_dir=args.logdir)
logger.info(f'Loaded config file: \'{args.config_file}\'')
logger.info(f'Running with configs:\n{cfg}')
logger.info(f'Running with {num_gpus} GPUs')

# python -m torch.distributed.launch train_feature_cycleGAN_psmnet.py --summary-freq 1 --save-freq 1 --logdir ../train_10_3_feature_cyclegan_psmnet/debug --debug
# python -m torch.distributed.launch train_feature_cycleGAN_psmnet.py --config-file configs/remote_train_steps.yaml --summary-freq 10 --save-freq 100 --logdir ../train_9_7_cyclegan_psmnet/debug --debug


def train(transformer_model, psmnet_model, transformer_optimizer, psmnet_optimizer,
          TrainImgLoader, ValImgLoader, attack):
    cur_err = np.inf    # store best result

    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars_psmnet = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (len(TrainImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE * num_gpus
            if global_step > cfg.SOLVER.STEPS:
                break

            # Adjust learning rate
            adjust_learning_rate(transformer_optimizer, global_step, cfg.SOLVER.LR_CASCADE, cfg.SOLVER.LR_STEPS)
            adjust_learning_rate(psmnet_optimizer, global_step, cfg.SOLVER.LR_CASCADE, cfg.SOLVER.LR_STEPS)

            do_summary = global_step % args.summary_freq == 0
            # Train one sample
            scalar_outputs_reproj, scalar_outputs_psmnet, \
            img_outputs_psmnet, img_output_reproj = \
                train_sample(sample, transformer_model, psmnet_model,
                             transformer_optimizer, psmnet_optimizer, attack, isTrain=True, adv_train=True)
            # Save result to tensorboard
            if (not is_distributed) or (dist.get_rank() == 0):

                scalar_outputs_psmnet = tensor2float(scalar_outputs_psmnet)
                avg_train_scalars_psmnet.update(scalar_outputs_psmnet)
                if do_summary:

                    # Update reprojection images
                    save_images_grid(summary_writer, 'train_reproj', img_output_reproj, global_step, nrow=3)
                    save_scalars(summary_writer, 'train_reproj', scalar_outputs_reproj, global_step)
                    # Update PSMNet images
                    save_images(summary_writer, 'train_psmnet', img_outputs_psmnet, global_step)
                    # Update PSMNet losses
                    scalar_outputs_psmnet.update({'lr': psmnet_optimizer.param_groups[0]['lr']})
                    save_scalars(summary_writer, 'train_psmnet', scalar_outputs_psmnet, global_step)


                # Save checkpoints
                if (global_step) % args.save_freq == 0:
                    checkpoint_data = {
                        'epoch': epoch_idx,
                        'Transformer': transformer_model.state_dict(),
                        'PSMNet': psmnet_model.state_dict(),
                        'optimizerTransformer': transformer_optimizer.state_dict(),
                        'optimizerPSMNet': psmnet_optimizer.state_dict()
                    }
                    save_filename = os.path.join(args.logdir, 'models', f'model_{global_step}.pth')
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric_psmnet = avg_train_scalars_psmnet.mean()
                    logger.info(f'Step {global_step} train psmnet: {total_err_metric_psmnet}')
        gc.collect()


def train_sample(sample, transformer_model, psmnet_model,
                 transformer_optimizer, psmnet_optimizer, attack, isTrain=True, adv_train=True):
    if isTrain:
        transformer_model.train()
        psmnet_model.train()
    else:
        transformer_model.eval()
        psmnet_model.eval()

    # Load data
    img_L = sample['img_L'].to(cuda_device)  # [bs, 3, H, W]
    img_R = sample['img_R'].to(cuda_device)  # [bs, 3, H, W]
    img_real_L = sample['img_real_L'].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample['img_real_R'].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_L = F.interpolate(img_real_L, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)
    img_real_R = F.interpolate(img_real_R, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)

    # Train on simple Transformer
    img_L_transformed, img_R_transformed, img_real_L_transformed, img_real_R_transformed \
        = transformer_model(img_L, img_R, img_real_L, img_real_R)  # [bs, 3, H, W]

    # Train on PSMNet
    disp_gt = sample['img_disp_l'].to(cuda_device)
    depth_gt = sample['img_depth_l'].to(cuda_device)  # [bs, 1, H, W]
    img_focal_length = sample['focal_length'].to(cuda_device)
    img_baseline = sample['baseline'].to(cuda_device)

    # Resize the 2x resolution disp and depth back to H * W
    # Note this should go before apply_disparity_cu
    disp_gt = F.interpolate(disp_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]
    depth_gt = F.interpolate(depth_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]

    if args.warp_op:
        img_disp_r = sample['img_disp_r'].to(cuda_device)
        img_disp_r = F.interpolate(img_disp_r, scale_factor=0.5, mode='nearest',
                                   recompute_scale_factor=False)
        disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        del img_disp_r

    # Get stereo loss on sim
    mask = (disp_gt < cfg.ARGS.MAX_DISP) * (disp_gt > 0)  # Note in training we do not exclude bg
    mask.detach()
    sim_pred_disp = 0
    if isTrain:
        _, _, pred_disp3 = psmnet_model(img_L, img_R, img_L_transformed, img_R_transformed)
        sim_pred_disp = pred_disp3

    # Get reprojection loss on sim
    sim_img_reproj_loss, sim_img_warped, sim_img_reproj_mask = get_reprojection_error(img_L, img_R, sim_pred_disp, mask)
    # sim_img_transformed_reproj_loss, sim_img_transformed_warped = get_reprojection_error(img_L_transformed, img_R_transformed, sim_pred_disp)

    adv_L, adv_R = 0, 0
    loss_psmnet = 0
    if isTrain and adv_train:
        input_L_warped = apply_disparity(img_R, -sim_pred_disp) # to do in the future get input_R_warped
        adv_L, adv_R, adv_LT, adv_RT  = attack.perturb(img_L, img_R, img_L_transformed, img_R_transformed, disp_gt, input_L_warped, input_L_warped, mask)
        pred_disp1, pred_disp2, pred_disp3 = psmnet_model(adv_L, adv_R, adv_LT, adv_RT)
        pred_disp = pred_disp3
        loss_psmnet = 0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt[mask], reduction='mean') \
               + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt[mask], reduction='mean') \
               + F.smooth_l1_loss(pred_disp3[mask], disp_gt[mask], reduction='mean')

    elif isTrain and not adv_train:
        pred_disp1, pred_disp2, pred_disp3 = psmnet_model(img_L, img_R, img_L_transformed, img_R_transformed)
        pred_disp = pred_disp3
        loss_psmnet = 0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt[mask], reduction='mean') \
               + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt[mask], reduction='mean') \
               + F.smooth_l1_loss(pred_disp3[mask], disp_gt[mask], reduction='mean')

    # Backward on sim
    # sim_loss_reproj = (sim_img_reproj_loss + sim_img_transformed_reproj_loss * 0.1) * 0.0001
    # sim_loss = (loss_psmnet + sim_loss_reproj) / 2
    sim_loss = loss_psmnet + sim_img_reproj_loss
    if isTrain:
        transformer_optimizer.zero_grad()
        psmnet_optimizer.zero_grad()
        sim_loss.backward()
        psmnet_optimizer.step()
        transformer_optimizer.step()

    # Get reprojection loss on real
    img_L_transformed, img_R_transformed, img_real_L_transformed, img_real_R_transformed \
        = transformer_model(img_L, img_R, img_real_L, img_real_R)  # [bs, 3, H, W]
    if isTrain:
        pred_disp1, pred_disp2, pred_disp3 = psmnet_model(img_real_L, img_real_R, img_real_L_transformed, img_real_L_transformed)
        real_pred_disp = pred_disp3
    else:
        with torch.no_grad():
            real_pred_disp = psmnet_model(img_real_L, img_real_R, img_real_L_transformed, img_real_L_transformed)
    real_img_reproj_loss, real_img_warped, real_img_reproj_mask = get_reprojection_error(img_real_L, img_real_R, real_pred_disp)
    # real_img_transformed_reproj_loss, real_img_transformed_warped = get_reprojection_error(img_real_L_transformed, img_real_R_transformed, real_pred_disp)

    # Backward on real
    # real_loss_reproj = (real_img_reproj_loss + real_img_transformed_reproj_loss * 0.1) * 0.0001
    # real_loss_reproj = real_img_reproj_loss / 2
    real_loss = real_img_reproj_loss
    if isTrain:
        transformer_optimizer.zero_grad()
        psmnet_optimizer.zero_grad()
        # real_loss_reproj.backward()
        real_loss.backward()
        psmnet_optimizer.step()
        transformer_optimizer.step()


    # Save reprojection outputs and images
    img_output_reproj = {
        'sim_img_reprojection': {
            'R': img_R, 'R_warped': sim_img_warped, 'mask': sim_img_reproj_mask
        },
        'real_img_reprojection': {
            'R': img_real_R, 'R_warped': real_img_warped, 'mask': real_img_reproj_mask
        }

    }
    scalar_outputs_reproj = {'sim_img_reproj_loss': sim_img_reproj_loss.item(),
                             'real_img_reproj_loss': real_img_reproj_loss.item()}

    # scalar_outputs_reproj = {'sim_img_reproj_loss': sim_img_reproj_loss.item(),
    #                          'sim_img_transformed_reproj_loss': sim_img_transformed_reproj_loss.item(),
    #                          'real_img_reproj_loss': real_img_reproj_loss.item(),
    #                          'real_img_transformed_reproj_loss': real_img_transformed_reproj_loss.item()}

    # Compute stereo error metrics on sim
    pred_disp = sim_pred_disp
    scalar_outputs_psmnet = {'loss': loss_psmnet.item()}
    scalar_outputs_psmnet.update(scalar_outputs_reproj)
    err_metrics = compute_err_metric(disp_gt,
                                     depth_gt,
                                     pred_disp,
                                     img_focal_length,
                                     img_baseline,
                                     mask)
    scalar_outputs_psmnet.update(err_metrics)
    # Compute error images
    pred_disp_err_np = disp_error_img(pred_disp[[0]], disp_gt[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2])))
    img_outputs_psmnet = {
        'disp_gt': disp_gt[[0]].repeat([1, 3, 1, 1]),
        'disp_pred': pred_disp[[0]].repeat([1, 3, 1, 1]),
        'disp_err': pred_disp_err_tensor,
        'adv_L': adv_L[[0]], # [B,3,H,W] or [0,1] or [0,255]
        'adv_LT': adv_LT[[0]],
        'img_L': img_L[[0]],
    }

    if is_distributed:
        scalar_outputs_psmnet = reduce_scalar_outputs(scalar_outputs_psmnet, cuda_device)
    return scalar_outputs_reproj, scalar_outputs_psmnet, img_outputs_psmnet, img_output_reproj


if __name__ == '__main__':
    # Obtain dataloader
    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, gaussian_blur=args.gaussian_blur, color_jitter=args.color_jitter,
                                    debug=args.debug, sub=100)
    val_dataset = MessytableDataset(cfg.SPLIT.VAL, gaussian_blur=args.gaussian_blur, color_jitter=args.color_jitter,
                                    debug=args.debug, sub=10)
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

    # Create Transformer model
    transformer_model = Transformer().to(cuda_device)
    transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=cfg.SOLVER.LR_CASCADE, betas=(0.9, 0.999))
    if is_distributed:
        transformer_model = torch.nn.parallel.DistributedDataParallel(
            transformer_model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        transformer_model = torch.nn.DataParallel(transformer_model)

    # Create PSMNet model
    psmnet_model = PSMNet(maxdisp=cfg.ARGS.MAX_DISP).to(cuda_device)
    psmnet_optimizer = torch.optim.Adam(psmnet_model.parameters(), lr=cfg.SOLVER.LR_CASCADE, betas=(0.9, 0.999))
    if is_distributed:
        psmnet_model = torch.nn.parallel.DistributedDataParallel(
            psmnet_model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        psmnet_model = torch.nn.DataParallel(psmnet_model)

    attack = FastGradientSignUntargeted(psmnet_model)
    # Start training
    train(transformer_model, psmnet_model, transformer_optimizer, psmnet_optimizer, TrainImgLoader, ValImgLoader, attack)
