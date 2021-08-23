"""
Author: Isabella Liu 8/16/21
Feature: Test cycleGAN model on sim-real dataset
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from nets.cycle_gan import CycleGANModel
from datasets.messytable_test import get_test_loader
from utils.config import cfg
from utils.util import get_time_string, setup_logger

parser = argparse.ArgumentParser(description='Testing for Cascade-Stereo on messy-table-dataset')
parser.add_argument('--config-file', type=str, default='./configs/local_test.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--model', type=str, default='', required=True, metavar='FILE', help='Path to test model')
parser.add_argument('--output', type=str, default='../testing_output_gan', help='Path to output folder')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
parser.add_argument('--annotate', type=str, default='', help='Annotation to the experiment')
parser.add_argument('--onreal', action='store_true', default=False, help='Test on real dataset')
parser.add_argument('--analyze-objects', action='store_true', default=True, help='Analyze on different objects')
parser.add_argument('--exclude-bg', action='store_true', default=False, help='Exclude background when testing')
parser.add_argument('--warp-op', action='store_true', default=True, help='Use warp_op function to get disparity')
parser.add_argument("--local_rank", type=int, default=0, help='Rank of device in distributed training')
args = parser.parse_args()
cfg.merge_from_file(args.config_file)
cuda_device = torch.device("cuda:{}".format(args.local_rank))

# python test_cycleGAN.py --model /code/models/model_4.pth --exclude-bg
# python test_cycleGAN.py --config-file configs/remote_test.yaml --model ../train_8_14_cascade/train1/models/model_best.pth --debug


def test(gan_model, val_loader, logger, log_dir):
    gan_model.eval()

    for iteration, data in enumerate(tqdm(val_loader)):
        img_L = data['img_L'].cuda()    # [bs, 1, H, W]
        img_R = data['img_R'].cuda()
        prefix = data['prefix'][0]

        img_L_real = data['img_L_real'].cuda()    # [bs, 1, H, W]
        img_L_real = F.interpolate(img_L_real, (540, 960), mode='bilinear',
                              recompute_scale_factor=False, align_corners=False)
        input_sample = {'img_L': img_L, 'img_R': img_R, 'img_real': img_L_real}
        gan_model.set_input(input_sample)
        with torch.no_grad():
            gan_model.forward()
            gan_model.compute_loss_G()

        # Save images
        img_outputs = {
            'img_L': {
                'input': gan_model.real_A_L, 'fake': gan_model.fake_B_L, 'rec': gan_model.rec_A_L,
                'idt': gan_model.idt_B_L
            },
            'img_R': {
                'input': gan_model.real_A_R, 'fake': gan_model.fake_B_R, 'rec': gan_model.rec_A_R,
                'idt': gan_model.idt_B_R
            },
            'img_Real': {
                'input': gan_model.real_B, 'fake': gan_model.fake_A, 'rec': gan_model.rec_B, 'idt': gan_model.idt_A
            }
        }
        img_list = []
        for tag, dict_value in img_outputs.items():
            for subtag, img_value in dict_value.items():
                img_list += [img_value[0]]
        img_grid = vutils.make_grid(img_list, padding=0, nrow=4, normalize=True, scale_each=True)
        img_grid_np = img_grid.permute(1, 2, 0).cpu().detach().numpy()
        plt.imsave(os.path.join(log_dir, f'{prefix}.png'), img_grid_np)
        plt.close('all')


def main():
    # Obtain the dataloader
    val_loader = get_test_loader(cfg.SPLIT.VAL, args.debug, sub=10, onReal=args.onreal)

    # Tensorboard and logger
    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f'{get_time_string()}_{args.annotate}')
    os.mkdir(log_dir)
    logger = setup_logger("CycleGAN Testing", distributed_rank=0, save_dir=log_dir)
    logger.info(f'Annotation: {args.annotate}')
    logger.info(f'Input args {args}')
    logger.info(f'Loaded config file \'{args.config_file}\'')
    logger.info(f'Running with configs:\n{cfg}')


    # Get GAN model
    gan_model = CycleGANModel()
    gan_model.set_device(cuda_device)
    gan_model.load_model(args.model)

    test(gan_model, val_loader, logger, log_dir)


if __name__ == '__main__':
    main()