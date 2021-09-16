"""
Author: Isabella Liu 8/15/21
Feature:
"""

import os
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from utils.config import cfg
from utils.util import load_pickle


def calc_left_ir_depth_from_rgb(k_main, k_l, rt_main, rt_l, rgb_depth):
    rt_lmain = rt_l @ np.linalg.inv(rt_main)
    h, w = rgb_depth.shape
    irl_depth = cv2.rgbd.registerDepth(k_main, k_l, None, rt_lmain, rgb_depth, (w, h), depthDilation=True)
    irl_depth[np.isnan(irl_depth)] = 0
    irl_depth[np.isinf(irl_depth)] = 0
    irl_depth[irl_depth < 0] = 0
    return irl_depth


class MessytableTestDataset(Dataset):
    def __init__(self, split_file, debug=False, sub=100, onReal=False):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_L, self.img_R, self.img_L_real, self.img_R_real, self.img_depth_l, self.img_depth_r, \
        self.img_meta, self.img_label, self.img_sim_realsense, self.img_real_realsense \
            = self.__get_split_files__(split_file, debug=debug, sub=sub)
        self.onReal = onReal

    @staticmethod
    def __get_split_files__(split_file, debug=False, sub=100):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :return: Lists of paths to the entries listed in split file
        """
        sim_dataset = cfg.DIR.DATASET
        real_dataset = cfg.REAL.DATASET
        sim_img_left_name = cfg.SPLIT.LEFT
        sim_img_right_name = cfg.SPLIT.RIGHT
        real_img_left_name = cfg.REAL.LEFT
        real_img_right_name = cfg.REAL.RIGHT
        sim_realsense = cfg.SPLIT.SIM_REALSENSE
        real_realsense = cfg.SPLIT.REAL_REALSENSE

        with open(split_file, 'r') as f:
            prefix = [line.strip() for line in f]

            img_L_sim = [os.path.join(sim_dataset, p, sim_img_left_name) for p in prefix]
            img_R_sim = [os.path.join(sim_dataset, p, sim_img_right_name) for p in prefix]
            img_L_real = [os.path.join(real_dataset, p, real_img_left_name) for p in prefix]
            img_R_real = [os.path.join(real_dataset, p, real_img_right_name) for p in prefix]
            img_depth_l = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHL) for p in prefix]
            img_depth_r = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHR) for p in prefix]
            img_meta = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.META) for p in prefix]
            img_label = [os.path.join(cfg.REAL.DATASET, p, cfg.SPLIT.LABEL) for p in prefix]
            img_sim_realsense = [os.path.join(sim_dataset, p, sim_realsense) for p in prefix]
            img_real_realsense = [os.path.join(real_dataset, p, real_realsense) for p in prefix]

            if debug is True:
                img_L_sim = img_L_sim[:sub]
                img_R_sim = img_R_sim[:sub]
                img_L_real = img_L_real[:sub]
                img_R_real = img_R_real[:sub]
                img_depth_l = img_depth_l[:sub]
                img_depth_r = img_depth_r[:sub]
                img_meta = img_meta[:sub]
                img_label = img_label[:sub]
                img_sim_realsense = img_sim_realsense[:sub]
                img_real_realsense = img_real_realsense[:sub]

        return img_L_sim, img_R_sim, img_L_real, img_R_real, img_depth_l, img_depth_r, img_meta, img_label, \
            img_sim_realsense, img_real_realsense

    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, idx):
        if self.onReal:
            img_L_rgb = (np.array(Image.open(self.img_L_real[idx]).convert(mode='L'))[:, :, None] - 127.5) / 127.5
            img_R_rgb = (np.array(Image.open(self.img_R_real[idx]).convert(mode='L'))[:, :, None] - 127.5) / 127.5
            img_depth_realsense = np.array(Image.open(self.img_real_realsense[idx])) / 1000
        else:
            img_L_rgb = (np.array(Image.open(self.img_L[idx]))[:, :, :1] - 127.5) / 127.5
            img_R_rgb = (np.array(Image.open(self.img_R[idx]))[:, :, :1] - 127.5) / 127.5
            img_L_rgb_real = (np.array(Image.open(self.img_L_real[idx]).convert(mode='L'))[:, :, None] - 127.5) / 127.5
            img_R_rgb_real = (np.array(Image.open(self.img_R_real[idx]).convert(mode='L'))[:, :, None] - 127.5) / 127.5
            img_depth_realsense = np.array(Image.open(self.img_sim_realsense[idx])) / 1000

        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000  # convert from mm to m
        img_depth_r = np.array(Image.open(self.img_depth_r[idx])) / 1000  # convert from mm to m
        img_meta = load_pickle(self.img_meta[idx])
        img_label = np.array(Image.open(self.img_label[idx]))

        # Convert depth map to disparity map
        extrinsic = img_meta['extrinsic']
        extrinsic_l = img_meta['extrinsic_l']
        extrinsic_r = img_meta['extrinsic_r']
        intrinsic = img_meta['intrinsic']
        intrinsic_l = img_meta['intrinsic_l']
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        # Convert img_depth_realsense to irL camera frame
        img_depth_realsense = calc_left_ir_depth_from_rgb(intrinsic, intrinsic_l,
                                                          extrinsic, extrinsic_l, img_depth_realsense)

        item = {}
        item['img_L'] = torch.tensor(img_L_rgb, dtype=torch.float32).permute(2, 0, 1)  # [bs, 1, H, W]
        item['img_R'] = torch.tensor(img_R_rgb, dtype=torch.float32).permute(2, 0, 1)  # [bs, 1, H, W]
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r'] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r'] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_realsense'] = torch.tensor(img_depth_realsense, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_label'] = torch.tensor(img_label, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = self.img_L[idx].split('/')[-2]
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if self.onReal is False:
            item['img_L_real'] = torch.tensor(img_L_rgb_real, dtype=torch.float32).permute(2, 0, 1)  # [bs, 1, H, W]
            item['img_R_real'] = torch.tensor(img_R_rgb_real, dtype=torch.float32).permute(2, 0, 1)  # [bs, 1, H, W]

        return item


def get_test_loader(split_file, debug=False, sub=100, onReal=False):
    """
    :param split_file: split file
    :param debug: Whether on debug mode, load less data
    :param sub: If on debug mode, how many items to load into dataset
    :param isTest: Whether on test, if test no random crop on input image
    :param onReal: Whether test on real dataset, folder and file name are different
    :return: dataloader
    """
    messytable_dataset = MessytableTestDataset(split_file, debug, sub, onReal=onReal)
    loader = DataLoader(messytable_dataset, batch_size=1, num_workers=0)
    return loader


if __name__ == '__main__':
    cdataset = MessytableTestDataset('/code/dataset_local_v9/training_lists/all.txt', onReal=True)
    item = cdataset.__getitem__(0)
    print(item['img_L'].shape)
    print(item['img_R'].shape)
    # print(item['img_L_real'].shape)
    print(item['img_disp_l'].shape)
    print(item['prefix'])
    print(item['img_depth_realsense'].shape)
