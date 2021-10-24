"""
Author: Isabella Liu 10/5/21
Feature:
"""

import torch
import torch.nn.functional as F
from .warp_ops import apply_disparity_cu


def apply_disparity(img, disp):
    batch_size, _, height, width = img.size()
    disp = disp / width

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                 width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                           padding_mode='zeros')

    return output


def get_reprojection_error(input_L, input_R, pred_disp_l, pred_disp_r, mask_l=None, mask_r=None):
    """
    input - [bs, c, h, w], feature or image
    pred_disp - [bs, 1, h, w]
    mask - [bs, 1, h, w]
    Note: apply_disparity use pred_disp_l to warp right image to left image (since F.grid_sample behaves a bit different),
    while appliy_disparity_cu use pred_disp_l to warp left to right
    """
    input_L_warped = apply_disparity(input_R, -pred_disp_l)
    input_R_warped = apply_disparity(input_L, pred_disp_r)
    if mask_l is None: # real does not have gt and thus mask
        # mask = torch.ones_like(input_L_warped).type(torch.bool)
        disp_gt_l = apply_disparity_cu(pred_disp_r, pred_disp_r.type(torch.int))  # [bs, 1, H, W]
        disp_gt_r = apply_disparity_cu(pred_disp_l, -pred_disp_l.type(torch.int))  # [bs, 1, H, W]
        mask_l = (disp_gt_l < 192) * (disp_gt_l > 0)  # Note in training we do not exclude bg
        mask_l = mask_l.detach()
        mask_r = (disp_gt_r < 192) * (disp_gt_r > 0)  # Note in training we do not exclude bg
        mask_r = mask_r.detach()
    bs, c, h, w = input_L.shape
    mask_l = mask_l.repeat(1, c, 1, 1)
    mask_r = mask_r.repeat(1, c, 1, 1)
    reprojection_loss_l = F.mse_loss(input_L_warped[mask_l], input_L[mask_l])
    reprojection_loss_r = F.mse_loss(input_R_warped[mask_r], input_R[mask_r])
    return reprojection_loss_l, reprojection_loss_r, input_L_warped, input_R_warped, \
           mask_l.type(torch.int), mask_r.type(torch.int)


def get_reprojection_error_old(input_L, input_R, pred_disp_l, mask=None):
    """
    input - [bs, c, h, w], feature or image
    pred_disp - [bs, 1, h, w], this should come from left camera frame
    mask - [bs, 1, h, w]
    Note: apply_disparity use pred_disp_l to warp right image to left image (since F.grid_sample behaves a bit different),
    while appliy_disparity_cu use pred_disp_l to warp left to right
    """
    input_L_warped = apply_disparity(input_R, -pred_disp_l)
    if mask is not None:
        bs, c, h, w = input_L.shape
        mask = mask.repeat(1, c, 1, 1)
    else:
        mask = torch.ones_like(input_L_warped).type(torch.bool)
    reprojection_loss = F.mse_loss(input_L_warped[mask], input_L[mask])
    return reprojection_loss, input_L_warped, mask.type(torch.int)


def get_reprojection_error_diff_ratio(input_L, input_R, pred_disp_l, mask=None):
    """
    input - [bs, c, h, w], feature or image
    pred_disp - [bs, 1, h, w], this should come from left camera frame
    mask - [bs, 1, h, w]
    Note: apply_disparity use pred_disp_l to warp right image to left image (since F.grid_sample behaves a bit different),
    while appliy_disparity_cu use pred_disp_l to warp left to right
    """
    ratio = [0.25, 0.5, 1]
    weight = [0.3, 0.5, 0.2]

    if mask is not None:
        bs, c, h, w = input_L.shape
        mask = mask.repeat(1, c, 1, 1)
    else:
        mask = torch.ones_like(input_L)
    mask = mask.type(torch.float32)
    mask.detach_()

    output = {}
    loss_dict = {}
    total_loss = 0
    for i, (r, w) in enumerate(zip(ratio, weight)):
        input_L_rs = F.interpolate(input_L, scale_factor=r, mode='bilinear')
        input_R_rs = F.interpolate(input_R, scale_factor=r, mode='bilinear')
        pred_disp_l_rs = F.interpolate(pred_disp_l, scale_factor=r, mode='bilinear') * r  # Note
        mask_rs = F.interpolate(mask, scale_factor=r, mode='bilinear').type(torch.bool)
        input_L_rs_warped = apply_disparity(input_R_rs, -pred_disp_l_rs)
        reproj_loss = F.mse_loss(input_L_rs_warped[mask_rs], input_L_rs[mask_rs])
        output.update({
            f'stage{i}':
                {
                    'target': input_L_rs,
                    'warped': input_L_rs_warped,
                    'pred_disp': pred_disp_l_rs,
                    'mask': mask_rs.type(torch.int)
                }
        })
        loss_dict.update({
            f'stage{i}': reproj_loss.item()
        })
        total_loss += reproj_loss * w
    return total_loss, output, loss_dict


if __name__ == '__main__':
    img_L = torch.rand(1, 1, 256, 512).cuda()
    img_R = torch.rand(1, 1, 256, 512).cuda()
    pred_disp = torch.rand(1, 1, 256, 512).cuda()
    loss, output = get_reprojection_error_diff_ratio(img_L, img_R, pred_disp)
    print(loss)
    print(len(output.keys()))
    print(output['stage1']['reproj_loss'])