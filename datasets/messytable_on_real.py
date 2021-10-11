"""
Author: Isabella Liu 9/27/21
Feature: Load data from messy-table-dataset, for training on real dataset
"""

import os
import numpy as np
import random
import cv2
from PIL import Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset

from utils.config import cfg
from utils.util import load_pickle


def __data_augmentation__(gaussian_blur=False, color_jitter=False):
    """
    :param gaussian_blur: Whether apply gaussian blur in data augmentation
    :param color_jitter: Whether apply color jitter in data augmentation
    Note:
        If you want to change the parameters of each augmentation, you need to go to config files,
        e.g. configs/remote_train_config.yaml
    """
    transform_list = [
        Transforms.ToTensor()
    ]
    if gaussian_blur:
        gaussian_sig = random.uniform(cfg.DATA_AUG.GAUSSIAN_MIN, cfg.DATA_AUG.GAUSSIAN_MAX)
        transform_list += [
            Transforms.GaussianBlur(kernel_size=cfg.DATA_AUG.GAUSSIAN_KERNEL, sigma=gaussian_sig)
        ]
    if color_jitter:
        bright = random.uniform(cfg.DATA_AUG.BRIGHT_MIN, cfg.DATA_AUG.BRIGHT_MAX)
        contrast = random.uniform(cfg.DATA_AUG.CONTRAST_MIN, cfg.DATA_AUG.CONTRAST_MAX)
        transform_list += [
            Transforms.ColorJitter(brightness=[bright, bright],
                                   contrast=[contrast, contrast])
        ]
    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation


class MessytableOnRealDataset(Dataset):
    def __init__(self, split_file, gaussian_blur=False, color_jitter=False, debug=False, sub=100, onreal=False):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param gaussian_blur: Whether apply gaussian blur in data augmentation
        :param color_jitter: Whether apply color jitter in data augmentation
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_L, self.img_R, self.img_depth_l, self.img_depth_r, self.img_meta, self.img_label = \
            self.__get_split_files__(split_file, debug, sub, isTest=False, onReal=onreal)
        self.gaussian_blur = gaussian_blur
        self.color_jitter = color_jitter
        self.onreal = onreal

    @staticmethod
    def __get_split_files__(split_file, debug=False, sub=100, isTest=False, onReal=False):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :param isTest: Whether on test, if test no random shuffle
        :param onReal: Whether test on real dataset, folder and file names are different
        :return: Lists of paths to the entries listed in split file
        """
        dataset = cfg.REAL.DATASET if onReal else cfg.DIR.DATASET
        img_left_name = cfg.REAL.LEFT if onReal else cfg.SPLIT.LEFT
        img_right_name = cfg.REAL.RIGHT if onReal else cfg.SPLIT.RIGHT

        with open(split_file, 'r') as f:
            prefix = [line.strip() for line in f]
            if isTest is False:
                np.random.shuffle(prefix)

            img_L = [os.path.join(dataset, p, img_left_name) for p in prefix]
            img_R = [os.path.join(dataset, p, img_right_name) for p in prefix]
            img_depth_l = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHL) for p in prefix]
            img_depth_r = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHR) for p in prefix]
            img_meta = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.META) for p in prefix]
            img_label = [os.path.join(cfg.REAL.DATASET, p, cfg.SPLIT.LABEL) for p in prefix]

            if debug is True:
                img_L = img_L[:sub]
                img_R = img_R[:sub]
                img_depth_l = img_depth_l[:sub]
                img_depth_r = img_depth_r[:sub]
                img_meta = img_meta[:sub]
                img_label = img_label[:sub]

        return img_L, img_R, img_depth_l, img_depth_r, img_meta, img_label

        # # If training with pix2pix + stereo, load sim dataset as input to the discriminator
        # if isTest is False:
        #     img_real_list = os.listdir(cfg.REAL.DATASET)
        #     img_real_list = [folder_name for folder_name in img_real_list if folder_name[0] in ['0', '1']]
        #     img_real_L = [os.path.join(cfg.REAL.DATASET, folder_name, cfg.REAL.LEFT) for folder_name in img_real_list]
        #     img_real_R = [os.path.join(cfg.REAL.DATASET, folder_name, cfg.REAL.RIGHT) for folder_name in img_real_list]
        #     img_real = img_real_L + img_real_R
        #     np.random.shuffle(img_real)
        #     return img_L, img_R, img_depth_l, img_depth_r, img_meta, img_label, img_real
        # else:
        #     return img_L, img_R, img_depth_l, img_depth_r, img_meta, img_label

    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, idx):
        if self.onreal:
            img_L_rgb = cv2.resize(cv2.imread(self.img_L[idx], flags=cv2.IMREAD_COLOR),
                                   (960, 540), interpolation=cv2.INTER_LINEAR)
            img_R_rgb = cv2.resize(cv2.imread(self.img_R[idx], flags=cv2.IMREAD_COLOR),
                                   (960, 540), interpolation=cv2.INTER_LINEAR)
            # img_L_rgb = ((np.array(img_L_rgb) - 127.5) / 127.5)[:, :, None]   # [H, W, 1], in (-1, 1)
            # img_R_rgb = ((np.array(img_R_rgb) - 127.5) / 127.5)[:, :, None]   # [H, W, 1], in (-1, 1)
        else:
            img_L_rgb = np.array(Image.open(self.img_L[idx]))[:, :, :-1]  # [H, W, 3]
            img_R_rgb = np.array(Image.open(self.img_R[idx]))[:, :, :-1]  # [H, W, 3]

        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000    # convert from mm to m
        img_depth_r = np.array(Image.open(self.img_depth_r[idx])) / 1000    # convert from mm to m
        img_meta = load_pickle(self.img_meta[idx])

        # # For unpaired pix2pix, load a random real image from real dataset [H, W, 1], in value range (-1, 1)
        # img_real_rgb = (np.array(Image.open(random.choice(self.img_real)))[:, :, None] - 127.5) / 127.5

        # Convert depth map to disparity map
        extrinsic_l = img_meta['extrinsic_l']
        extrinsic_r = img_meta['extrinsic_r']
        intrinsic_l = img_meta['intrinsic_l']
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        # random crop the image to 256 * 512
        h, w = img_L_rgb.shape[:2]
        th, tw = cfg.ARGS.CROP_HEIGHT, cfg.ARGS.CROP_WIDTH
        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)
        img_L_rgb = img_L_rgb[x:(x+th), y:(y+tw)]
        img_R_rgb = img_R_rgb[x:(x+th), y:(y+tw)]
        img_disp_l = img_disp_l[2*x: 2*(x+th), 2*y: 2*(y+tw)]  # depth original res in 1080*1920
        img_depth_l = img_depth_l[2*x: 2*(x+th), 2*y: 2*(y+tw)]
        img_disp_r = img_disp_r[2*x: 2*(x+th), 2*y: 2*(y+tw)]
        img_depth_r = img_depth_r[2*x: 2*(x+th), 2*y: 2*(y+tw)]
        # img_real_rgb = img_real_rgb[2*x: 2*(x+th), 2*y: 2*(y+tw)]  # real original res in 1080*1920

        # Get data augmentation
        custom_augmentation = __data_augmentation__(gaussian_blur=False, color_jitter=False)

        item = {}
        item['img_L'] = custom_augmentation(img_L_rgb).type(torch.FloatTensor)  # [bs, 1, H, W]
        item['img_R'] = custom_augmentation(img_R_rgb).type(torch.FloatTensor)  # [bs, 1, H, W]
        # item['img_L'] = torch.tensor(img_L_rgb, dtype=torch.float32).permute(2, 0, 1)  # [bs, 1, H, W]
        # item['img_R'] = torch.tensor(img_R_rgb, dtype=torch.float32).permute(2, 0, 1)  # [bs, 1, H, W]

        # item['img_real'] = torch.tensor(img_real_rgb, dtype=torch.float32).permute(2, 0, 1)  # [bs, 3, 2*H, 2*W]
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W] in dataloader
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r'] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r'] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = self.img_L[idx].split('/')[-2]
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return item


if __name__ == '__main__':
    cdataset = MessytableOnRealDataset('/code/dataset_local_v9/training_lists/poseTrain.txt', onreal=True)
    item = cdataset.__getitem__(0)
    print(item['img_L'].shape)
    print(item['img_R'].shape)
    print(item['img_disp_l'].shape)
    print(item['prefix'])
