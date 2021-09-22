"""
Author: Isabella Liu 9/3/21
Feature: Load data for testing real sense
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from utils.config import cfg
from utils.util import load_pickle
from utils.test_util import calc_left_ir_depth_from_rgb


class MessytableRealSenseDataset(Dataset):
    def __init__(self, split_file, debug=False, sub=100):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_sim_realsense, self.img_real_realsense, self.img_depth, self.img_depth_l, self.img_depth_r, \
        self.img_meta, self.img_label = self.__get_split_files__(split_file, debug=debug, sub=sub)

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
        sim_realsense = cfg.SPLIT.SIM_REALSENSE
        real_realsense = cfg.SPLIT.REAL_REALSENSE

        with open(split_file, 'r') as f:
            prefix = [line.strip() for line in f]

            img_sim_realsense = [os.path.join(sim_dataset, p, sim_realsense) for p in prefix]
            img_real_realsense = [os.path.join(real_dataset, p, real_realsense) for p in prefix]
            img_depth = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTH) for p in prefix]
            img_depth_l = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHL) for p in prefix]
            img_depth_r = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHR) for p in prefix]
            img_meta = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.META) for p in prefix]
            img_label = [os.path.join(cfg.REAL.DATASET, p, cfg.SPLIT.LABEL) for p in prefix]

            if debug is True:
                img_sim_realsense = img_sim_realsense[:sub]
                img_real_realsense = img_real_realsense[:sub]
                img_depth = img_depth[:sub]
                img_depth_l = img_depth_l[:sub]
                img_depth_r = img_depth_r[:sub]
                img_meta = img_meta[:sub]
                img_label = img_label[:sub]

        return img_sim_realsense, img_real_realsense, img_depth, img_depth_l, img_depth_r, img_meta, img_label

    def __len__(self):
        return len(self.img_sim_realsense)

    def __getitem__(self, idx):
        img_sim_realsense = np.array(Image.open(self.img_sim_realsense[idx])) / 1000  # convert from mm to m
        img_real_realsense = np.array(Image.open(self.img_real_realsense[idx])) / 1000  # convert from mm to m
        img_depth = np.array(Image.open(self.img_depth[idx])) / 1000  # convert from mm to m
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
        focal_length = intrinsic_l[0, 0]

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]
        mask = img_depth > 0
        img_disp = np.zeros_like(img_depth)
        img_disp[mask] = focal_length * baseline / img_depth[mask]

        # Convert realsense_depth from rgb frame to irL frame
        img_sim_realsense = calc_left_ir_depth_from_rgb(intrinsic, intrinsic_l,
                                                        extrinsic, extrinsic_l, img_sim_realsense)
        img_real_realsense = calc_left_ir_depth_from_rgb(intrinsic, intrinsic_l,
                                                         extrinsic, extrinsic_l, img_real_realsense)

        item = {}
        item['img_sim_realsense'] = torch.tensor(img_sim_realsense, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_real_realsense'] = torch.tensor(img_real_realsense, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp'] = torch.tensor(img_disp, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth'] = torch.tensor(img_depth, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r'] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r'] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_label'] = torch.tensor(img_label, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = self.img_sim_realsense[idx].split('/')[-2]
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return item


def get_realsense_loader(split_file, debug=False, sub=100):
    """
    :param split_file: split file
    :param debug: Whether on debug mode, load less data
    :param sub: If on debug mode, how many items to load into dataset
    :param isTest: Whether on test, if test no random crop on input image
    :param onReal: Whether test on real dataset, folder and file name are different
    :return: dataloader
    """
    messytable_dataset = MessytableRealSenseDataset(split_file, debug, sub)
    loader = DataLoader(messytable_dataset, batch_size=1, num_workers=0)
    return loader


if __name__ == '__main__':
    cdataset = MessytableRealSenseDataset('/code/dataset_local_v9/training_lists/all.txt')
    item = cdataset.__getitem__(0)
    print(item['img_sim_realsense'].shape)
    print(item['img_real_realsense'].shape)
    print(item['img_depth'].shape)
    print(item['img_disp'].shape)
    print(item['prefix'])
