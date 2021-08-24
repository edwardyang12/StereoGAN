import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines

class CustomDataset(Dataset):
    def __init__(self, datapath, list_filename):
        self.datapath = datapath
        self.left_filenames, self.right_filenames = self.load_path(list_filename)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]

        return left_images, right_images

    def load_image(self, filename):
        img = np.array(Image.open(filename).convert('L')).astype(np.float32)
        # img = cv2.GaussianBlur(img,(9, 9),0.1,2)
        return Image.fromarray(img.astype(np.uint8))

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        # right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        processed = get_transform()
        left_img = processed(left_img).numpy()
        # right_img = processed(right_img).numpy()


        return left_img, os.path.join(self.datapath, self.left_filenames[index])
