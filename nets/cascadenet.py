"""
Author: Isabella Liu 8/7/21
Feature: Cascade-stereo net based on PSMNet, uses multiple stages doing PSMNet on different resolution
Reference: Most part comes from https://github.com/alibaba/cascade-stereo/blob/master/CasStereoNet/models/psmnet.py
"""

import math

import torch

from .psmnet_submodule import *

Align_Corners = False
Align_Corners_Range = False


def disparity_regression(x, disp_values):
    assert len(x.shape) == 4
    return torch.sum(x * disp_values, 1, keepdim=False)


def get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns=False, max_disp=192.0):
    # shape, (B, H, W)
    # cur_disp: (B, H, W)
    # return disp_range_samples: (B, D, H, W)
    if not using_ns:
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel)  # (B, H, W)
        cur_disp_max = (cur_disp + ndisp / 2 * disp_inteval_pixel)
        # cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel).clamp(min=0.0)   #(B, H, W)
        # cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_inteval_pixel).clamp(max=max_disp)

        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                       dtype=cur_disp.dtype,
                                                                       requires_grad=False).reshape(1, -1, 1, 1)
                                                          * new_interval.unsqueeze(1))
    else:
        # using neighbor region information to help determine the range.
        # consider the maximum and minimum values ​​in the region.
        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
        B, H, W = cur_disp.shape
        cur_disp_smooth = F.interpolate((cur_disp / 4.0).unsqueeze(1),
                                        [H // 4, W // 4], mode='bilinear', align_corners=Align_Corners_Range).squeeze(1)
        # get minimum value
        disp_min_ns = torch.abs(
            F.max_pool2d(-cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2))  # (B, 1/4H, 1/4W)
        # get maximum value
        disp_max_ns = F.max_pool2d(cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2)

        disp_pred_inter = torch.abs(disp_max_ns - disp_min_ns)  # (B, 1/4H, 1/4W)
        disp_range_comp = (ndisp // 4 * disp_inteval_pixel - disp_pred_inter).clamp(min=0) / 2.0  # (B, 1/4H, 1/4W)

        cur_disp_min = (disp_min_ns - disp_range_comp).clamp(min=0, max=max_disp)
        cur_disp_max = (disp_max_ns + disp_range_comp).clamp(min=0, max=max_disp)

        new_interval = (cur_disp_max - cur_disp_min) / (ndisp // 4 - 1)  # (B, 1/4H, 1/4W)

        # (B, 1/4D, 1/4H, 1/4W)
        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp // 4, device=cur_disp.device,
                                                                       dtype=cur_disp.dtype,
                                                                       requires_grad=False).reshape(1, -1, 1, 1)
                                                          * new_interval.unsqueeze(1))
        # (B, D, H, W)
        disp_range_samples = F.interpolate((disp_range_samples * 4.0).unsqueeze(1),
                                           [ndisp, H, W],
                                           mode='trilinear',
                                           align_corners=Align_Corners_Range).squeeze(1)
    return disp_range_samples


def get_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, device, dtype, shape, using_ns, ns_size,
                           max_disp=192.0):
    # shape, (B, H, W)
    # cur_disp: (B, H, W) or float
    # return disp_range_values: (B, D, H, W)
    # with torch.no_grad():
    if cur_disp is None:
        cur_disp = torch.tensor(0, device=device, dtype=dtype, requires_grad=False).reshape(1, 1, 1).repeat(*shape)
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel).clamp(min=0.0)  # (B, H, W)
        cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_inteval_pixel).clamp(max=max_disp)
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

        disp_range_volume = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1, 1)
                                                         * new_interval.unsqueeze(1))
    else:
        disp_range_volume = get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns,
                                                       max_disp)

    return disp_range_volume


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class feature_extraction(nn.Module):
    def __init__(self, arch_mode="nospp", num_stage=None, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        assert arch_mode in ["spp", "nospp"]
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.inplanes = 32
        self.concat_feature_channel = concat_feature_channel
        # strid 1
        self.firstconv_a = nn.Sequential(convbn(1, 32, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         convbn(32, 32, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True))
        # strid 2
        self.firstconv_b = nn.Sequential(convbn(32, 32, 3, 2, 1, 1),
                                         nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)

        # strid 4
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.out1_cat = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, self.concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                bias=False))

        self.out_channels = [(320, self.concat_feature_channel)]

        if self.arch_mode == 'spp':
            raise NotImplementedError

        elif self.arch_mode == "nospp":
            final_chs = 320
            if num_stage == 3:
                self.inner1 = nn.Conv2d(32, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(32, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, 160, 3, padding=1, bias=False)
                self.out2_cat = nn.Conv2d(160, self.concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                          bias=False)

                self.out3 = nn.Conv2d(final_chs, 80, 3, padding=1, bias=False)
                self.out3_cat = nn.Conv2d(80, self.concat_feature_channel // 4, kernel_size=1, padding=0, stride=1,
                                          bias=False)

                self.out_channels.append((160, self.concat_feature_channel // 2))
                self.out_channels.append((80, self.concat_feature_channel // 4))

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(32, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, 160, 3, padding=1, bias=False)
                self.out2_cat = nn.Conv2d(160, self.concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                          bias=False)

                self.out_channels.append((160, self.concat_feature_channel // 2))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output_s1 = self.firstconv_a(x)
        output = self.firstconv_b(output_s1)
        output_s2 = self.layer1(output)
        l2 = self.layer2(output_s2)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        output_msfeat = {}

        output_feature = torch.cat((l2, l3, l4), dim=1)
        out = output_feature
        out_cat = self.out1_cat(output_feature)
        output_msfeat["stage1"] = {"gwc_feature": out, "concat_feature": out_cat}

        intra_feat = output_feature

        if self.arch_mode == "nospp":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(output_s2)
                out = self.out2(intra_feat)
                out_cat = self.out2_cat(out)
                output_msfeat["stage2"] = {"gwc_feature": out, "concat_feature": out_cat}

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(output_s1)
                out = self.out3(intra_feat)
                out_cat = self.out3_cat(out)
                output_msfeat["stage3"] = {"gwc_feature": out, "concat_feature": out_cat}

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(output_s2)
                out = self.out2(intra_feat)
                out_cat = self.out2_cat(out)
                output_msfeat["stage2"] = {"gwc_feature": out, "concat_feature": out_cat}

        return output_msfeat


class CostAggregation(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(CostAggregation, self).__init__()

        self.dres0 = nn.Sequential(convbn_3d(in_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(base_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(base_channels, base_channels, 3, 1, 1))

        self.dres2 = hourglass(base_channels)

        self.dres3 = hourglass(base_channels)

        self.dres4 = hourglass(base_channels)

        self.classif0 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, cost, FineD, FineH, FineW, disp_range_samples):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        cost3 = self.classif3(out3)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost0 = F.interpolate(cost0, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)
            cost1 = F.interpolate(cost1, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)
            cost2 = F.interpolate(cost2, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)

            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, disp_range_samples)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, disp_range_samples)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, disp_range_samples)

        cost3 = F.interpolate(cost3, [FineD, FineH, FineW], mode='trilinear', align_corners=Align_Corners)
        cost3 = torch.squeeze(cost3, 1)
        pred3_prob = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparity_regression(pred3_prob, disp_range_samples)

        if self.training:
            return pred0, pred1, pred2, pred3
        else:
            return pred3


class GetCostVolume(nn.Module):
    def __init__(self):
        super(GetCostVolume, self).__init__()

    def get_warped_feats(self, x, y, disp_range_samples, ndisp):
        bs, channels, height, width = y.size()

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
        mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
        mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

        cur_disp_coords_y = mh
        cur_disp_coords_x = mw - disp_range_samples

        # print("cur_disp", cur_disp, cur_disp.shape if not isinstance(cur_disp, float) else 0)
        # print("cur_disp_coords_x", cur_disp_coords_x, cur_disp_coords_x.shape)

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4)  # (B, D, H, W, 2)

        # (B, C, D, H, W)
        y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2),
                                 mode='bilinear',
                                 padding_mode='zeros',
                                 align_corners=Align_Corners).view(bs, channels, ndisp, height, width)

        # a little difference, no zeros filling
        x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1)  # (B, C, D, H, W)
        x_warped = x_warped.transpose(0, 1)  # (C, B, D, H, W)
        # x1 = x2 + d >= d
        x_warped[:, mw < disp_range_samples] = 0
        x_warped = x_warped.transpose(0, 1)  # (B, C, D, H, W)

        return x_warped, y_warped

    def build_concat_volume(self, x, y, disp_range_samples, ndisp):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        concat_cost = x.new().resize_(bs, channels * 2, ndisp, height, width).zero_()  # (B, 2C, D, H, W)

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp)
        concat_cost[:, x.size()[1]:, :, :, :] = y_warped
        concat_cost[:, :x.size()[1], :, :, :] = x_warped

        return concat_cost

    def build_gwc_volume(self, x, y, disp_range_samples, ndisp, num_groups):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp)  # (B, C, D, H, W)

        assert channels % num_groups == 0
        channels_per_group = channels // num_groups
        gwc_cost = (x_warped * y_warped).view([bs, num_groups, channels_per_group, ndisp, height, width]).mean(
            dim=2)  # (B, G, D, H, W)

        return gwc_cost

    def forward(self, features_left, features_right, disp_range_samples, ndisp, num_groups):
        # bs, channels, height, width = features_left["gwc_feature"].size()
        gwc_volume = self.build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"],
                                           disp_range_samples, ndisp, num_groups)

        concat_volume = self.build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                 disp_range_samples, ndisp)

        volume = torch.cat((gwc_volume, concat_volume), 1)  # (B, C+G, D, H, W)

        return volume


class CascadeNet(nn.Module):
    def __init__(self, maxdisp, ndisps, disp_interval_pixel, using_ns, ns_size, grad_method="detach",
                 cr_base_chs=[32, 32, 32]):
        super(CascadeNet, self).__init__()
        self.maxdisp = maxdisp
        self.ndisps = ndisps
        self.disp_interval_pixel = disp_interval_pixel
        self.num_stage = len(self.ndisps)
        self.cr_base_chs = cr_base_chs
        self.grad_method = grad_method
        self.ns_size = ns_size
        self.using_ns = using_ns
        self.num_groups = [40, 20, 10]
        self.concat_channels = 12
        assert self.maxdisp == 192
        assert self.grad_method in ["detach", "undetach"]

        self.stage_infos = {
            "stage1": {
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }
        # print(
        #     "***********ndisps:{}  disp_interval_pixel:{} grad:{} ns:{} ns_size:{} cr_base_chs:{} ************".format(
        #         self.ndisps, self.disp_interval_pixel, self.grad_method, self.using_ns, self.ns_size, self.cr_base_chs))

        self.feature_extraction = feature_extraction(num_stage=self.num_stage, arch_mode="nospp",
                                                     concat_feature_channel=self.concat_channels)

        self.get_cv = GetCostVolume()

        cr_feats_in_chs = [self.num_groups[i] + chs1 * 2 for i, (chs0, chs1) in
                           enumerate(self.feature_extraction.out_channels)]

        self.cost_agg = nn.ModuleList([CostAggregation(in_channels=cr_feats_in_chs[i], base_channels=cr_base_chs[i])
                                       for i in range(self.num_stage)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        """
        :param left: [bs, 1, H, W]
        :param right: [bs, 1, H, W]
        :return: Predicted disparity
        """

        refimg_msfea = self.feature_extraction(left)
        targetimg_msfea = self.feature_extraction(right)

        outputs = {}
        pred, cur_disp = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            if pred is not None:
                if self.grad_method == "detach":
                    cur_disp = pred.detach()
                else:
                    cur_disp = pred
            disp_range_samples = get_disp_range_samples(cur_disp=cur_disp,
                                                        ndisp=self.ndisps[stage_idx],
                                                        disp_inteval_pixel=self.disp_interval_pixel[stage_idx],
                                                        dtype=left.dtype,
                                                        device=left.device,
                                                        shape=[left.shape[0], left.shape[2], left.shape[3]],
                                                        max_disp=self.maxdisp,
                                                        using_ns=self.using_ns,
                                                        ns_size=self.ns_size)

            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            refimg_fea, targetimg_fea = refimg_msfea["stage{}".format(stage_idx + 1)], \
                                        targetimg_msfea["stage{}".format(stage_idx + 1)]

            # matching
            cost = self.get_cv(refimg_fea, targetimg_fea,
                               disp_range_samples=F.interpolate((disp_range_samples / stage_scale).unsqueeze(1),
                                                                [self.ndisps[stage_idx] // int(stage_scale),
                                                                 left.size()[2] // int(stage_scale),
                                                                 left.size()[3] // int(stage_scale)],
                                                                mode='trilinear',
                                                                align_corners=Align_Corners_Range).squeeze(1),
                               ndisp=self.ndisps[stage_idx] // int(stage_scale),
                               num_groups=self.num_groups[stage_idx])

            if self.training:
                pred0, pred1, pred2, pred3 = self.cost_agg[stage_idx](cost,
                                                                      FineD=self.ndisps[stage_idx],
                                                                      FineH=left.shape[2],
                                                                      FineW=left.shape[3],
                                                                      disp_range_samples=disp_range_samples)
                pred = pred3
                outputs_stage = {
                    "pred0": pred0,
                    "pred1": pred1,
                    "pred2": pred2,
                    "pred3": pred3,
                    "pred": pred}
                outputs["stage{}".format(stage_idx + 1)] = outputs_stage
                outputs.update(outputs_stage)

            else:
                pred3 = self.cost_agg[stage_idx](cost,
                                                 FineD=self.ndisps[stage_idx],
                                                 FineH=left.shape[2],
                                                 FineW=left.shape[3],
                                                 disp_range_samples=disp_range_samples)
                pred = pred3
                outputs_stage = {
                    "pred3": pred3,
                    "pred": pred}
                outputs["stage{}".format(stage_idx + 1)] = outputs_stage

        return outputs


def cascade_loss(inputs, target, mask, **kwargs):
    """
    :param inputs: Dict of outputs
    :param target: [bs, H, W]
    :param mask: [bs, H, W]
    :param kwargs:
    :return: Cascade stereo loss
    """
    disp_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=target.dtype, device=target.device, requires_grad=False)
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        disp0, disp1, disp2, disp3 = stage_inputs["pred0"], stage_inputs["pred1"], stage_inputs["pred2"], stage_inputs["pred3"]

        loss = 0.5 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + \
               0.5 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + \
               0.7 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean') + \
               1.0 * F.smooth_l1_loss(disp3[mask], target[mask], reduction='mean')

        if disp_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += disp_loss_weights[stage_idx] * loss
        else:
            total_loss += loss
    return total_loss


if __name__ == '__main__':
    cascade_model = CascadeNet(
        maxdisp=192,
        ndisps=[48, 24],
        disp_interval_pixel=[4.0, 1.0],
        cr_base_chs=[32, 32, 16],
        grad_method='detach',
        using_ns=True,
        ns_size=3
    ).cuda()
    cascade_model.train()

    # Input image
    img_left = torch.rand(1, 3, 256, 512).cuda()
    img_right = torch.rand(1, 3, 256, 512).cuda()
    outputs = cascade_model(img_left, img_right)
    print(outputs.keys())

    # Test cascade loss
    mask = torch.ones(1, 256, 512).type(torch.bool).cuda()  # [3, H, W]
    target = torch.rand(1, 256, 512).cuda()
    loss = cascade_loss(outputs, target, mask, dlossw=[0.5, 2.0])
    print(loss)