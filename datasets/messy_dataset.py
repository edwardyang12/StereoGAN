import os
import random
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import numpy as np
from datasets.data_io import get_transform_train, get_transform_test, get_transform_img, read_all_lines
import pickle
from datasets.warp_ops import *
import torch
import torchvision.transforms as transforms
from datasets.warp_ops import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MESSYDataset(Dataset):
    def __init__(self, datapath, list_filename, training, crop_width, crop_height, test_crop_width, test_crop_height, left_img, right_img, args):
        self.datapath = datapath
        self.training = training
        self.depthpath = args.depthpath
        self.left_img = left_img
        self.right_img = right_img
        self.args = args
        self.left_filenames, self.right_filenames, self.disp_filenames_L, self.disp_filenames_R, self.disp_filenames, self.meta_filenames = self.load_path(list_filename)

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.test_crop_width = test_crop_width
        self.test_crop_height = test_crop_height


        if self.training:
            assert self.disp_filenames_L is not None
            assert self.disp_filenames_R is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)

        left_images = [os.path.join(x,self.left_img) for x in lines]
        right_images = [os.path.join(x,self.right_img) for x in lines]

        #label_images = [os.path.join(x,"label.png") for x in lines]
        disp_images_L = [os.path.join(x,"depthL.png") for x in lines]
        disp_images_R = [os.path.join(x,"depthR.png") for x in lines]
        disp_images = [os.path.join(x,"depth.png") for x in lines]
        meta = [os.path.join(x,"meta.pkl") for x in lines]
        return left_images, right_images, disp_images_L, disp_images_R, disp_images, meta


    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


    def load_image(self, filename, half):
        img = Image.open(filename).convert('L')
        if half:
            img = img.resize((int(img.size[0]/2),int(img.size[1]/2)))
        return img

    def load_label(self, filename, half):
        img = Image.open(filename).convert('I;16')
        if half:
            img = img.resize((int(img.size[0]/2),int(img.size[1]/2)), resample=Image.NEAREST)
        return img

    def load_disp(self, filename_L, filename_R, filename, metafile):
        img_L = Image.open(filename_L)
        img_R = Image.open(filename_R)
        img = Image.open(filename)
        meta = self.load_pickle(metafile)

        img_L = img_L.resize((int(img_L.size[0]/2),int(img_L.size[1]/2)))
        img_R = img_R.resize((int(img_R.size[0]/2),int(img_R.size[1]/2)))
        img = img.resize((int(img.size[0]/2),int(img.size[1]/2)))
        data_L = np.asarray(img_L,dtype=np.float32)
        data_R = np.asarray(img_R,dtype=np.float32)
        data = np.asarray(img,dtype=np.float32)

        data_L_mask = (data_L < 0)
        data_R_mask = (data_R < 0)
        data_mask = (data < 0)
        data_L[data_L_mask] = 0.0
        data_R[data_R_mask] = 0.0
        data[data_mask] = 0.0
        #if not (torch.all(torch.tensor(data_L) >= 0) and torch.all(torch.tensor(data_R) >= 0)):
        #    print("neg found " + filename_L + ", " + filename_R)
        #print(meta)
        e = meta['extrinsic'][:3,3]
        el = meta['extrinsic_l'][:3,3]
        er = meta['extrinsic_r'][:3,3]

        b = np.linalg.norm(el-er)*1000
        br = np.linalg.norm(el-e)*1000
        f = meta['intrinsic_r'][0][0]/2
        frgb = meta['intrinsic'][0][0]/2

        mask_l = (data_L == 0)
        mask_r = (data_R == 0)
        mask = (data == 0)
        dis_L = b*f/data_L
        dis_L[mask_l] = 0
        dis_R = b*f/data_R
        dis_R[mask_r] = 0
        dis_rgb = br*frgb/data
        dis_rgb[mask] = 0
        return b, br, f, data_L, data_R, data, dis_L, dis_R, dis_rgb

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]), self.left_img == "1024_irL_real_1080.png")
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]), self.left_img == "1024_irL_real_1080.png")
        #label = self.load_label(os.path.join(self.datapath, self.label[index]), True)




        if self.disp_filenames_L:  # has disparity ground truth
            if self.left_img == "1024_irL_real_1080.png":
                path = self.depthpath
            else:
                path = self.datapath
            b, br, f, depthL, depthR, depth, disparity_L, disparity_R, disparity = self.load_disp(os.path.join(path, self.disp_filenames_L[index]), \
                                                    os.path.join(path, self.disp_filenames_R[index]), \
                                                    os.path.join(path, self.disp_filenames[index]), \
                                                    os.path.join(path, self.meta_filenames[index]))
            #print(type(disparity_R), disparity_R.shape)
            #disparity_R_t = torch.tensor(disparity_R)
            #disparity_R_ti = torch.tensor(disparity_R, dtype=torch.int)
            #disparity_R_t = disparity_R_t.reshape((1,1,disparity_R_t.shape[0],disparity_R_t.shape[1]))
            #disparity_R_ti = disparity_R_ti.reshape((1,1,disparity_R_ti.shape[0],disparity_R_ti.shape[1]))
            #disparity_L_from_R = apply_disparity_cu(disparity_R_t, disparity_R_ti)

        else:
            disparity_L = None
            disparity_R = None
            #disparity_L_from_R = None

        if self.training:
            #print("left_img: ", left_img.size, " right_img: ", right_img.size, " dis_gt: ", disparity.size)
            w, h = left_img.size
            crop_w, crop_h = self.crop_width, self.crop_height

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity_L = disparity_L[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_R = disparity_R[y1:y1 + crop_h, x1:x1 + crop_w]


            # to tensor, normalize
            color_jitter = transforms.ColorJitter(brightness=float(self.args.brightness), contrast=float(self.args.contrast), saturation=0, hue=0)

            processed = get_transform_train(color_jitter, self.args.kernel, tuple([float(e) for e in self.args.var.split(",") if e]), self.args.use_blur, self.args.use_jitter)

            left_img = processed(left_img)
            if self.args.diff_jitter:
                color_jitter = transforms.ColorJitter(brightness=float(self.args.brightness), contrast=float(self.args.contrast), saturation=0, hue=0)
                processed = get_transform_train(color_jitter, self.args.kernel, tuple([float(e) for e in self.args.var.split(",") if e]), self.args.use_blur, self.args.use_jitter)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity_R}
        else:
            #print(torch.all(torch.tensor(disparity_R) >= 0))
            w, h = left_img.size

            # normalize
            ##color_jitter = transforms.ColorJitter(brightness=0, contrast=0, saturation=0)
            processed = get_transform_test()
            processedimg = get_transform_img()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()
            #print("before", np.array(label))


            #label = processedimg(label).numpy()


            #print("after", label)
            # pad to size 1248x384
            top_pad = self.test_crop_height - h
            right_pad = self.test_crop_width - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity_R is not None:
                assert len(disparity_R.shape) == 2
                disparity_R = np.lib.pad(disparity_R, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity_R is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity_R,
                        "disparity_rgb": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "depth": depthL,
                        "baseline": b,
                        "baseline_rgb": br,
                        "f": f}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
