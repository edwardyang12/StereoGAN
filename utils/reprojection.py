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


def get_reprojection_error(input_L, input_R, pred_disp, gt_disp=None):
    """
    input - [bs, c, h, w], feature or image
    pred_disp - [bs, 1, h, w], this should come from left camera frame
    mask - [bs, 1, h, w]
    """
    input_R_warped = apply_disparity(input_L, pred_disp)
    if gt_disp is not None:
        disp_warped = apply_disparity_cu(gt_disp, -gt_disp.type(torch.int))
    else:
        disp_warped = apply_disparity_cu(pred_disp, -pred_disp.type(torch.int))
    mask = (disp_warped < 192) * (disp_warped > 0)
    mask = mask.repeat(1, 3, 1, 1)
    reprojection_loss = F.mse_loss(input_R[mask], input_R_warped[mask])
    return reprojection_loss, input_R_warped, mask.type(torch.int)


if __name__ == '__main__':
    img_L = torch.rand(1, 3, 256, 512).cuda()
    img_R = torch.rand(1, 3, 256, 512).cuda()
    pred_disp = torch.rand(1, 1, 256, 512).cuda()
    # img_L.requires_grad = True
    # img_R.requires_grad = True
    pred_disp.requires_grad = True
    # loss, _ = get_reprojection_error(img_L, img_R, pred_disp)
    # print(pred_disp.grad)
    # # loss.backward()
    # print(loss)

    # loss2 = F.mse_loss(img_L, img_R)
    img_L2 = img_L * 2
    print(img_L2.grad)