import numpy as np
import re
import torchvision.transforms as transforms


def get_transform_train(color_jitter, kernel, var, use_blur, use_jitter):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #print(color_jitter.brightness)
    _, b, c, _, _ = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation, color_jitter.hue)
    if use_blur and use_jitter:
        return transforms.Compose([
            transforms.ColorJitter(brightness=(b,b), contrast=(c,c), saturation=0, hue=0),
            transforms.GaussianBlur(kernel,var),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        if use_blur:
            return transforms.Compose([
                transforms.GaussianBlur(kernel,var),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif use_jitter:
            return transforms.Compose([
                transforms.ColorJitter(brightness=(b,b), contrast=(c,c), saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])


def get_transform_test():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_transform_img():
    return transforms.Compose([
        transforms.ToTensor()
    ])


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
