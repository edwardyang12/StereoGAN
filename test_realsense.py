"""
Author: Isabella Liu 9/3/21
Feature: Test real sense
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from datasets.messytable_realsense import get_realsense_loader
from utils.cascade_metrics import compute_err_metric, compute_obj_err
from utils.config import cfg
from utils.util import get_time_string, setup_logger, depth_error_img, disp_error_img
from utils.test_util import load_from_dataparallel_model, save_img, save_obj_err_file
from utils.warp_ops import apply_disparity_cu

parser = argparse.ArgumentParser(description='Testing for Cascade-Stereo on messy-table-dataset')
parser.add_argument('--config-file', type=str, default='./configs/local_test.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--output', type=str, default='../testing_output_realsense', help='Path to output folder')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
parser.add_argument('--annotate', type=str, default='', help='Annotation to the experiment')
parser.add_argument('--onreal', action='store_true', default=False, help='Compute real realsense')
parser.add_argument('--analyze-objects', action='store_true', default=True, help='Analyze on different objects')
parser.add_argument('--exclude-bg', action='store_true', default=False, help='Exclude background when testing')
parser.add_argument('--warp-op', action='store_true', default=True, help='Use warp_op function to get disparity')
parser.add_argument('--count-zeros', action='store_true', default=False, help='Whether count zeros in pred depth')
args = parser.parse_args()
cfg.merge_from_file(args.config_file)


# python test_realsense.py --onreal --exclude-bg
# python test_realsense.py --config-file configs/remote_test.yaml --onreal --exclude-bg --debug


def test(val_loader, logger, log_dir):
    total_err_metrics = {'epe': 0, 'bad1': 0, 'bad2': 0,
                         'depth_abs_err': 0, 'depth_err2': 0, 'depth_err4': 0, 'depth_err8': 0}
    total_obj_disp_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_4_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_count = np.zeros(cfg.SPLIT.OBJ_NUM)
    os.mkdir(os.path.join(log_dir, 'pred_disp'))
    os.mkdir(os.path.join(log_dir, 'gt_disp'))
    os.mkdir(os.path.join(log_dir, 'pred_disp_abs_err_cmap'))
    os.mkdir(os.path.join(log_dir, 'pred_depth'))
    os.mkdir(os.path.join(log_dir, 'gt_depth'))
    os.mkdir(os.path.join(log_dir, 'pred_depth_abs_err_cmap'))

    for iteration, data in enumerate(tqdm(val_loader)):
        if args.onreal:
            pred_depth = data['img_real_realsense'].cuda()
        else:
            pred_depth = data['img_sim_realsense'].cuda()

        img_disp_l = data['img_disp_l'].cuda()
        img_depth_l = data['img_depth_l'].cuda()
        img_label = data['img_label'].cuda()
        img_focal_length = data['focal_length'].cuda()
        img_baseline = data['baseline'].cuda()
        prefix = data['prefix'][0]

        # If using warp_op, computing img_disp_l from img_disp_r
        if args.warp_op:
            img_disp_r = data['img_disp_r'].cuda()
            img_depth_r = data['img_depth_r'].cuda()
            img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
            img_depth_l = apply_disparity_cu(img_depth_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]

        # Resize all the depths to [540, 960]
        pred_depth = F.interpolate(pred_depth, (540, 960), mode='nearest',
                                   recompute_scale_factor=False)
        img_disp_l = F.interpolate(img_disp_l, (540, 960), mode='nearest',
                                   recompute_scale_factor=False)
        img_depth_l = F.interpolate(img_depth_l, (540, 960), mode='nearest',
                                    recompute_scale_factor=False)
        img_label = F.interpolate(img_label, (540, 960), mode='nearest',
                                  recompute_scale_factor=False).type(torch.int)

        if args.exclude_bg:
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            # TODO some pred_depth has 0 value in non-ground area
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * img_ground_mask if args.count_zeros else \
                (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * (pred_depth > 0) * img_ground_mask
        else:
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) if args.count_zeros else \
                (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * (pred_depth > 0)
        mask = mask.type(torch.bool)  # [bs, 1, H, W]

        ground_mask = torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()

        # Some pixels in pred depth are 0
        nonzero_mask = pred_depth > 0
        pred_disp = torch.zeros_like(pred_depth)
        pred_disp[nonzero_mask] = img_focal_length * img_baseline / pred_depth[nonzero_mask]

        # Get loss metric
        err_metrics = compute_err_metric(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                         img_baseline, mask, pred_depth)
        for k in total_err_metrics.keys():
            total_err_metrics[k] += err_metrics[k]
        logger.info(f'Test instance {prefix} - {err_metrics}')

        # Get object error
        obj_disp_err, obj_depth_err, \
            obj_depth_4_err, obj_count = compute_obj_err(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                                     img_baseline, img_label, mask, cfg.SPLIT.OBJ_NUM)
        total_obj_disp_err += obj_disp_err
        total_obj_depth_err += obj_depth_err
        total_obj_depth_4_err += obj_depth_4_err
        total_obj_count += obj_count

        # Get disparity image
        pred_disp_np = pred_disp.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
        pred_disp_np[ground_mask] = -1

        # Get disparity ground truth image
        gt_disp_np = img_disp_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_disp_np[ground_mask] = -1

        # Get disparity error image
        pred_disp_err_np = disp_error_img(pred_disp, img_disp_l, mask)

        # Get depth image
        pred_depth_np = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()  # in m, [H, W]
        # crop depth map to [0.2m, 2m]
        # pred_depth_np[pred_depth_np < 0.2] = -1
        # pred_depth_np[pred_depth_np > 2] = -1
        pred_depth_np[ground_mask] = -1

        # Get depth ground truth image
        gt_depth_np = img_depth_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_depth_np[ground_mask] = -1

        # Get depth error image
        pred_depth_err_np = depth_error_img(pred_depth * 1000, img_depth_l * 1000, mask)

        # Save images
        save_img(log_dir, prefix, pred_disp_np, gt_disp_np, pred_disp_err_np,
                 pred_depth_np, gt_depth_np, pred_depth_err_np)

    # Get final error metrics
    for k in total_err_metrics.keys():
        total_err_metrics[k] /= len(val_loader)
    logger.info(f'\nTest on {len(val_loader)} instances\n {total_err_metrics}')

    # # Save object error to csv file
    total_obj_disp_err /= total_obj_count
    total_obj_depth_err /= total_obj_count
    total_obj_depth_4_err /= total_obj_count
    save_obj_err_file(total_obj_disp_err, total_obj_depth_err, total_obj_depth_4_err, log_dir)

    logger.info(f'Successfully saved object error to obj_err.txt')


def main():
    # Obtain the dataloader
    val_loader = get_realsense_loader(cfg.SPLIT.VAL, args.debug, sub=10)

    # Tensorboard and logger
    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f'{get_time_string()}_{args.annotate}')
    os.mkdir(log_dir)
    logger = setup_logger("RealSense Testing", distributed_rank=0, save_dir=log_dir)
    logger.info(f'Annotation: {args.annotate}')
    logger.info(f'Input args {args}')
    logger.info(f'Loaded config file \'{args.config_file}\'')
    logger.info(f'Running with configs:\n{cfg}')

    test(val_loader, logger, log_dir)


if __name__ == '__main__':
    main()
