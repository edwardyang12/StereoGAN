import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import pandas as pd
from math import inf
import cv2

class CustomDatasetTest(Dataset):
    def __init__(self, datapath, list_filename):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.meta_filenames, self.label_filenames = self.load_path(list_filename)
        self.crop_width = 64
        self.crop_height = 64

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]

        disp_images = [x[2] for x in splits]
        meta = [x[3] for x in splits]
        label = [x[4] for x in splits]

        return left_images, right_images, disp_images, meta, label

    def load_image(self, filename):
        img = np.array(Image.open(filename).convert('L')).astype(np.float32)
        # img = cv2.GaussianBlur(img,(9, 9),0.1,2)
        return (Image.fromarray(img.astype(np.uint8))).resize((480,270), resample=Image.NEAREST)

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        else:
            disparity = None

        if self.meta_filenames:
            temp = pd.read_pickle(os.path.join(self.datapath, self.meta_filenames[index]))
            intrinsic = temp['intrinsic']
            baseline = abs((temp['extrinsic_l']-temp['extrinsic_r'])[0][3])

            temp = disparity*256.

            temp = (baseline*1000*intrinsic[0][0]/2)/(temp)
            temp[temp==inf] = 0
            disparity = temp

        if self.label_filenames:
            temp = os.path.join(self.datapath, self.label_filenames[index])
            label = np.array(Image.open(temp).resize((960,540), resample=Image.NEAREST))


        w, h = left_img.size
        crop_w, crop_h = self.crop_width, self.crop_height

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

        # random crop
        left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        
        processed = get_transform()
        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()


        return {"left": left_img,
                "right": right_img,
                "disparity": disparity,
                "label": label,
                "intrinsic": intrinsic,
                "baseline": baseline}
