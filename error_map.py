from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils
import os

def gen_error_colormap(sim, real):
    cols = np.array(
        [[0, 0.5, 49, 54, 149],
         [0.5 , 1, 69, 117, 180],
         [1, 2, 116, 173, 209],
         [2, 4, 171, 217, 233],
         [4, 8, 224, 243, 248],
         [8, 16, 254, 224, 144],
         [16, 32, 253, 174, 97],
         [32, 64, 244, 109, 67],
         [64, 128, 215, 48, 39],
         [128, 256, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.

    H, W = sim.shape
    error = np.abs(sim - real)

    error_image = np.zeros([H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]

    for i in range(cols.shape[0]):
        distance = 20
        error_image[:10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return error_image

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

# from file to histogram
def readHistogram(data_filenames, data_dir, depth=False, label=False):
    lines = utils.read_text_lines(data_filenames)

    occur = dict()
    for line in lines:
        splits = line.split()
        mask = []
        left_img, right_img = splits[:2]
        sample = {}

        if(label):
            if 'left' not in sample:
                left =  os.path.join(data_dir, left_img)
                right = os.path.join(data_dir, right_img)
                sample['left'] = np.array(Image.open(left).convert('RGB')).astype(np.float32)
                sample['right'] =np.array(Image.open(right).convert('RGB')).astype(np.float32)
            temp = Image.fromarray(sample['left'].astype('uint8'))
            left = np.array(temp.convert('L'))

            mask = np.array(Image.open(os.path.join(data_dir, splits[4])).resize((960,540), resample=Image.NEAREST))
            left[mask>=17] = 0
            mask  = left>0.
            left = left[mask]

        if(depth):
            if 'left' not in sample:
                left =  os.path.join(data_dir, left_img)
                right = os.path.join(data_dir, right_img)
                sample['left'] = np.array(Image.open(left).convert('RGB')).astype(np.float32)
                sample['right'] =np.array(Image.open(right).convert('RGB')).astype(np.float32)
            #depthR so use right image
            temp = Image.fromarray(sample['right'].astype('uint8'))
            left = np.array(temp.convert('L'))

            mask = np.array(Image.open(os.path.join(data_dir, splits[2])))
            temp = (mask>0.) & (mask<2000.)
            left = left[temp]

        if(not depth and not label and not transforms and not gauss):
            left = np.array(Image.open(left).convert('L')).flatten()

        unique, counts = np.unique(left, return_counts=True)
        temp = dict(zip(unique, counts))
        occur = merge_two_dicts(occur, temp)

    return occur

if __name__ == "__main__":
    # difference between two images
    # a = np.array(Image.open('pictures/0-1-0/0128_irL_denoised_half.jpg').convert('L'))
    # b = np.array(Image.open('pictures/0-1-0/0128_irL_denoised_half.jpg').convert('L'))
    # b = cv2.GaussianBlur(a,(3, 3),0)

    # c = gen_error_colormap(b, a)

    # plt.imshow(c)
    # plt.show()

    data_filenames = 'filenames/custom_test_sim.txt'
    data_dir = 'linked_sim_v9'

    # occur = histogram(data_filenames, data_dir)
    occur = {0: 304427, 1: 324093, 2: 46932, 3: 41981, 4: 35657, 5: 33375, 6: 31560, 7: 30946, 8: 30395, 9: 30378, 10: 29953, 11: 29698, 12: 30020, 13: 29793, 14: 29629, 15: 30197, 16: 30227, 17: 30410, 18: 30825, 19: 30606, 20: 30041, 21: 30296, 22: 29380, 23: 28921, 24: 28875, 25: 28706, 26: 27983, 27: 27584, 28: 26677, 29: 26033, 30: 25405, 31: 24880, 32: 23763, 33: 22999, 34: 22279, 35: 21228, 36: 20468, 37: 19727, 38: 19048, 39: 17641, 40: 16497, 41: 15106, 42: 14038, 43: 12951, 44: 12314, 45: 11501, 46: 9815, 47: 7803, 48: 6314, 49: 4905, 50: 3587, 51: 2554, 52: 1758, 53: 1155, 54: 700, 55: 430, 56: 258, 57: 141, 58: 77, 59: 31, 60: 18, 61: 10, 62: 5, 63: 2, 64: 2, 65: 32, 66: 22, 67: 15, 68: 6, 69: 4, 70: 1}
    plt.bar(list(occur.keys()), occur.values(), color='g')
    plt.show()
