"""
Author: Isabella Liu 7/18/21
Feature: Load data from messy-table-dataset
"""

import os
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset
import cv2

from utils.config import cfg
from utils.util import load_pickle


def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


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

    # If training with pix2pix + cascade, load real dataset as input to the discriminator
    if isTest is False:
        img_real_list = os.listdir(cfg.REAL.DATASET)
        img_real_list = [folder_name for folder_name in img_real_list if folder_name[0] in ['0', '1']]
        img_real_L = [os.path.join(cfg.REAL.DATASET, folder_name, cfg.REAL.LEFT) for folder_name in img_real_list]
        img_real_R = [os.path.join(cfg.REAL.DATASET, folder_name, cfg.REAL.RIGHT) for folder_name in img_real_list]
        return img_L, img_R, img_depth_l, img_depth_r, img_meta, img_label, img_real_L, img_real_R
    else:
        return img_L, img_R, img_depth_l, img_depth_r, img_meta, img_label


class MessytableDataset(Dataset):
    def __init__(self, split_file, gaussian_blur=False, color_jitter=False, debug=False, sub=100):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param gaussian_blur: Whether apply gaussian blur in data augmentation
        :param color_jitter: Whether apply color jitter in data augmentation
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_L, self.img_R, self.img_depth_l, self.img_depth_r, self.img_meta, self.img_label, self.img_real_L,\
            self.img_real_R = __get_split_files__(split_file, debug, sub, isTest=False)
        self.gaussian_blur = gaussian_blur
        self.color_jitter = color_jitter
        self.real_len = len(self.img_real_L)

    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, idx):
        img_L_rgb = np.array(Image.open(self.img_L[idx]))[:, :, :-1]   # [H, W, 3]
        img_R_rgb = np.array(Image.open(self.img_R[idx]))[:, :, :-1]   # [H, W, 3]
        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000    # convert from mm to m
        img_depth_r = np.array(Image.open(self.img_depth_r[idx])) / 1000    # convert from mm to m
        img_meta = load_pickle(self.img_meta[idx])

        # Load real images
        real_idx = np.random.randint(0, high=self.real_len)
        img_real_L_rgb = Image.open(self.img_real_L[real_idx]).convert(mode='L')
        img_real_R_rgb = Image.open(self.img_real_R[real_idx]).convert(mode='L')
        img_real_L_rgb = np.array(img_real_L_rgb)
        img_real_R_rgb = np.array(img_real_R_rgb)
        # img_real_L_rgb = gamma_trans(np.array(img_real_L_rgb), 0.25)
        # img_real_R_rgb = gamma_trans(np.array(img_real_R_rgb), 0.25)
        img_real_L_rgb = np.repeat(img_real_L_rgb[:, :, None], 3, axis=-1)
        img_real_R_rgb = np.repeat(img_real_R_rgb[:, :, None], 3, axis=-1)

        # img_real_L_rgb = np.array(Image.open(self.img_real_L[real_idx]))[:, :, None]
        # img_real_L_rgb = np.repeat(img_real_L_rgb, 3, axis=-1)  # [H, W, 3]
        # img_real_R_rgb = np.array(Image.open(self.img_real_R[real_idx]))[:, :, None]
        # img_real_R_rgb = np.repeat(img_real_R_rgb, 3, axis=-1)  # [H, W, 3]

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
        img_real_L_rgb = img_real_L_rgb[2*x: 2*(x+th), 2*y: 2*(y+tw)]  # real original res in 1080*1920
        img_real_R_rgb = img_real_R_rgb[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]

        # Get data augmentation
        custom_augmentation = __data_augmentation__(gaussian_blur=self.gaussian_blur, color_jitter=self.color_jitter)
        normalization = __data_augmentation__(gaussian_blur=False, color_jitter=False)

        item = {}
        item['img_L'] = custom_augmentation(img_L_rgb).type(torch.FloatTensor)
        item['img_R'] = custom_augmentation(img_R_rgb).type(torch.FloatTensor)
        item['img_real_L'] = normalization(img_real_L_rgb).type(torch.FloatTensor)
        item['img_real_R'] = normalization(img_real_R_rgb).type(torch.FloatTensor)
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W] in dataloader
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r'] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r'] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = self.img_L[idx].split('/')[-2]
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return item


if __name__ == '__main__':
    cdataset = MessytableDataset(cfg.SPLIT.TRAIN)
    item = cdataset.__getitem__(0)
    print(item['img_L'].shape)
    print(item['img_R'].shape)
    print(item['img_disp_l'].shape)
    print(item['prefix'])
    print(item['img_real_L'].shape)
