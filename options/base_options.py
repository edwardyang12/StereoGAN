import argparse
import os
from utils import util
import torch
import nets as models
from datasets import __datasets__
from nets import __models__, __loss__
#import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='/cephfs/jianyu', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_6blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        #========================================================================================================================================================================

        parser.add_argument('--cmodel', default='gwcnet-c', help='select a model structure', choices=__models__.keys())
        parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

        parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
        parser.add_argument('--datapath', required=True, help='data path')
        parser.add_argument('--depthpath', required=True, help='depth path')
        parser.add_argument('--test_dataset', required=True, help='dataset name', choices=__datasets__.keys())
        parser.add_argument('--test_datapath', required=True, help='data path')
        parser.add_argument('--test_sim_datapath', required=True, help='data path')
        parser.add_argument('--test_real_datapath', required=True, help='data path')
        parser.add_argument('--trainlist', required=True, help='training list')
        parser.add_argument('--testlist', required=True, help='testing list')
        parser.add_argument('--sim_testlist', required=True, help='testing list')
        parser.add_argument('--real_testlist', required=True, help='testing list')

        parser.add_argument('--clr', type=float, default=0.001, help='base learning rate')
        parser.add_argument('--cbatch_size', type=int, default=1, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
        parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
        parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

        parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
        parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
        parser.add_argument('--resume', action='store_true', help='continue training the model')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

        parser.add_argument('--summary_freq', type=int, default=50, help='the frequency of saving summary')
        parser.add_argument('--test_summary_freq', type=int, default=50, help='the frequency of saving summary')
        parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

        parser.add_argument('--log_freq', type=int, default=50, help='log freq')
        parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
        parser.add_argument("--local_rank", type=int, default=0)

        parser.add_argument('--mode', type=str, default="train", help='train or test mode')


        parser.add_argument('--ndisps', type=str, default="48,24", help='ndisps')
        parser.add_argument('--disp_inter_r', type=str, default="4,1", help='disp_intervals_ratio')
        parser.add_argument('--dlossw', type=str, default="0.5,2.0", help='depth loss weight for different stage')
        parser.add_argument('--cr_base_chs', type=str, default="32,32,16", help='cost regularization base channels')
        parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='predicted disp detach, undetach')


        parser.add_argument('--using_ns', action='store_true', help='using neighbor search')
        parser.add_argument('--ns_size', type=int, default=3, help='nb_size')

        parser.add_argument('--crop_height', type=int, required=True, help="crop height")
        parser.add_argument('--crop_width', type=int, required=True, help="crop width")
        parser.add_argument('--test_crop_height', type=int, required=True, help="crop height")
        parser.add_argument('--test_crop_width', type=int, required=True, help="crop width")

        parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
        parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
        parser.add_argument('--opt-level', type=str, default="O0")
        parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
        parser.add_argument('--loss-scale', type=str, default=None)

        parser.add_argument('--feat_map', type=int, default=None)

        parser.add_argument('--use_jitter', action='store_true', help='use color jitter.')
        parser.add_argument('--use_blur', action='store_true', help='use gaussian blur.')
        parser.add_argument('--diff_jitter', action='store_true', help='use different color jitter on both images.')
        parser.add_argument('--brightness', type=str, default=None)
        parser.add_argument('--contrast', type=str, default=None)
        parser.add_argument('--kernel', type=int, default=None)
        parser.add_argument('--var', type=str, default=None)

        parser.add_argument('--simtosim', action='store_true', help='sim2sim training for gan.')

        parser.add_argument('--dcropsize', type=int, default=None)
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        opt.model = "cycle_gan"
        # modify model-related parser options
        print(opt.model)
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

    