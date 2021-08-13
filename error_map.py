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
    occur = {0: 308059, 1: 172029, 2: 136536, 3: 126565, 4: 124157, 5: 118650, 6: 115075, 7: 110959, 8: 108411, 9: 106375, 10: 107829, 11: 110128, 12: 113564, 13: 112241, 14: 109631, 15: 129097, 16: 224911, 17: 206210, 18: 149870, 19: 98339, 20: 69752, 21: 54464, 22: 44613, 23: 38444, 24: 35126, 25: 32351, 26: 30059, 27: 28236, 28: 26505, 29: 24853, 30: 23138, 31: 21776, 32: 19958, 33: 19191, 34: 17862, 35: 16842, 36: 15821, 37: 15504, 38: 14784, 39: 13656, 40: 13626, 41: 13092, 42: 11008, 43: 9960, 44: 9253, 45: 8710, 46: 8160, 47: 7738, 48: 7319, 49: 7077, 50: 6872, 51: 6558, 52: 6419, 53: 6078, 54: 6019, 55: 5840, 56: 5778, 57: 5656, 58: 5472, 59: 5267, 60: 5163, 61: 5192, 62: 5016, 63: 4902, 64: 4759, 65: 4651, 66: 4604, 67: 4307, 68: 4347, 69: 4213, 70: 3960, 71: 3957, 72: 3868, 73: 3771, 74: 3566, 75: 3451, 76: 3274, 77: 3217, 78: 2989, 79: 2828, 80: 2786, 81: 2643, 82: 2518, 83: 2336, 84: 2201, 85: 2008, 86: 1907, 87: 1866, 88: 1635, 89: 1597, 90: 1454, 91: 1375, 92: 1279, 93: 1168, 94: 1035, 95: 1046, 96: 925, 97: 859, 98: 786, 99: 720, 100: 642, 101: 559, 102: 567, 103: 476, 104: 464, 105: 425, 106: 369, 107: 347, 108: 318, 109: 327, 110: 266, 111: 236, 112: 209, 113: 182, 114: 165, 115: 166, 116: 148, 117: 126, 118: 109, 119: 113, 120: 84, 121: 98, 122: 84, 123: 64, 124: 66, 125: 48, 126: 52, 127: 48, 128: 42, 129: 41, 130: 33, 131: 22, 132: 20, 133: 20, 134: 24, 135: 27, 136: 16, 137: 16, 138: 11, 139: 15, 140: 10, 141: 16, 142: 14, 143: 10, 144: 6, 145: 6, 146: 5, 147: 4, 149: 1, 150: 3, 151: 3, 152: 1, 155: 1, 156: 4, 157: 1, 158: 2, 163: 4, 165: 2, 166: 1, 148: 4, 153: 3, 170: 1, 154: 6, 162: 1, 167: 1, 164: 1, 159: 1, 160: 2, 173: 1, 161: 1, 168: 1, 176: 1, 188: 1, 171: 1, 183: 1, 174: 1, 180: 1, 178: 1, 175: 1, 172: 2, 186: 1, 169: 1, 177: 1, 181: 1, 197: 1, 184: 1, 179: 1, 182: 1, 189: 1, 194: 1, 185: 1, 191: 1, 187: 2, 195: 1, 201: 2, 193: 1}
    plt.bar(list(occur.keys()), occur.values(), color='g')
    plt.show()
