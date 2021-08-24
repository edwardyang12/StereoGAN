import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from nets import __models__, __loss__


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt, model):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['s1_netG_A', 's1_netG_B', 's2_netG_A', 's2_netG_B', 's1_netD_A', 's1_netD_B', 's2_netD_A', 's2_netD_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.s1_netG_A = networks.define_G(opt.s1_input_nc, opt.s1_output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.s1_netG_B = networks.define_G(opt.s1_output_nc, opt.s1_input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.s2_netG_A = networks.define_G(opt.s2_input_nc, opt.s2_output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.s2_netG_B = networks.define_G(opt.s2_output_nc, opt.s2_input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.cascade = model
        self.cascade.train()

        self.model_loss = __loss__[opt.cmodel]

        self.opt = opt

        if self.isTrain:  # define discriminators
            self.s1_netD_A = networks.define_D(opt.s1_output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.s1_netD_B = networks.define_D(opt.s1_input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.s2_netD_A = networks.define_D(opt.s2_output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.s2_netD_B = networks.define_D(opt.s2_input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.s1_input_nc == opt.s1_output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.s1_optimizer_G = torch.optim.Adam(itertools.chain(self.s1_netG_A.parameters(), self.s1_netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.s1_optimizer_D = torch.optim.Adam(itertools.chain(self.s1_netD_A.parameters(), self.s1_netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.s1_optimizer_G)
            self.optimizers.append(self.s1_optimizer_D)

            self.s2_optimizer_G = torch.optim.Adam(itertools.chain(self.s2_netG_A.parameters(), self.s2_netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.s2_optimizer_D = torch.optim.Adam(itertools.chain(self.s2_netD_A.parameters(), self.s2_netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.s2_optimizer_G)
            self.optimizers.append(self.s2_optimizer_D)

            self.optimizer_cascade = torch.optim.Adam(self.cascade.parameters(), lr=opt.clr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_cascade)


    def set_input(self, s1_inputL_sim, s1_inputR_sim, s2_inputL_sim, s2_inputR_sim, s1_inputL_real, s1_inputR_real, s2_inputL_real, s2_inputR_real, real_gt):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        #AtoB = self.opt.direction == 'AtoB'
        self.s1_real_L = s1_inputL_real.to(self.device)
        self.s1_real_R = s1_inputR_real.to(self.device)

        self.s2_real_L = s2_inputL_real.to(self.device)
        self.s2_real_R = s2_inputR_real.to(self.device)

        self.s1_sim_L = s1_inputL_sim.to(self.device)
        self.s1_sim_R = s1_inputR_sim.to(self.device)

        self.s2_sim_L = s2_inputL_sim.to(self.device)
        self.s2_sim_R = s2_inputR_sim.to(self.device)

        self.real_gt = real_gt.to(self.device)

        self.mask = (self.real_gt < opt.maxdisp) & (self.real_gt > 0)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.s1_fake_B_sim_L = self.s1_netG_A(self.s1_real_L)  # G_A(A)
        self.s1_rec_A_real_L = self.s1_netG_B(self.s1_fake_B_sim_L)   # G_B(G_A(A))
        self.s1_fake_A_real_L = self.s1_netG_B(self.s1_sim_L)  # G_B(B)
        self.s1_rec_B_sim_L = self.s1_netG_A(self.s1_fake_A_real_L)   # G_A(G_B(B))

        self.s1_fake_B_sim_R = self.s1_netG_A(self.s1_real_R)  # G_A(A)
        self.s1_rec_A_real_R = self.s1_netG_B(self.s1_fake_B_sim_R)   # G_B(G_A(A))
        self.s1_fake_A_real_R = self.s1_netG_B(self.s1_sim_R)  # G_B(B)
        self.s1_rec_B_sim_R = self.s1_netG_A(self.s1_fake_A_real_R)   # G_A(G_B(B))

        self.s2_fake_B_sim_L = self.s2_netG_A(self.s2_real_L)  # G_A(A)
        self.s2_rec_A_real_L = self.s2_netG_B(self.s2_fake_B_sim_L)   # G_B(G_A(A))
        self.s2_fake_A_real_L = self.s2_netG_B(self.s2_sim_L)  # G_B(B)
        self.s2_rec_B_sim_L = self.s2_netG_A(self.s2_fake_A_real_L)   # G_A(G_B(B))

        self.s2_fake_B_sim_R = self.s2_netG_A(self.s2_real_R)  # G_A(A)
        self.s2_rec_A_real_R = self.s2_netG_B(self.s2_fake_B_sim_R)   # G_B(G_A(A))
        self.s2_fake_A_real_R = self.s2_netG_B(self.s2_sim_R)  # G_B(B)
        self.s2_rec_B_sim_R = self.s2_netG_A(self.s2_fake_A_real_R)   # G_A(G_B(B))

        self.cascade.set_gan_train(self.s1_fake_B_sim_L, self.s2_fake_B_sim_L, self.s1_fake_B_sim_R, self.s2_fake_B_sim_R)

        self.cs_outputs = self.cascade(self.s2_fake_B_sim_L, self.s2_fake_B_sim_R)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        s1_fake_B_L = self.fake_B_pool.query(self.s1_fake_B_sim_L)
        self.s1_loss_D_A_L = self.backward_D_basic(self.s1_netD_A, self.s1_sim_L, s1_fake_B_L)

        s2_fake_B_L = self.fake_B_pool.query(self.s2_fake_B_sim_L)
        self.s2_loss_D_A_L = self.backward_D_basic(self.s2_netD_A, self.s2_sim_L, s2_fake_B_L)

        s1_fake_B_R = self.fake_B_pool.query(self.s1_fake_B_sim_R)
        self.s1_loss_D_A_R = self.backward_D_basic(self.s1_netD_A, self.s1_sim_R, s1_fake_B_R)

        s2_fake_B_R = self.fake_B_pool.query(self.s2_fake_B_sim_R)
        self.s2_loss_D_A_R = self.backward_D_basic(self.s2_netD_A, self.s2_sim_R, s2_fake_B_R)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        s1_fake_A_L = self.fake_A_pool.query(self.s1_fake_A_real_L)
        self.s1_loss_D_B_L = self.backward_D_basic(self.s1_netD_B, self.s1_real_L, s1_fake_A_L)

        s2_fake_A_L = self.fake_A_pool.query(self.s2_fake_A_real_L)
        self.s2_loss_D_B_L = self.backward_D_basic(self.s2_netD_B, self.s2_real_L, s2_fake_A_L)

        s1_fake_A_R = self.fake_A_pool.query(self.s1_fake_A_real_R)
        self.s1_loss_D_B_R = self.backward_D_basic(self.s1_netD_B, self.s1_real_R, s1_fake_A_R)

        s2_fake_A_R = self.fake_A_pool.query(self.s2_fake_A_real_L)
        self.s2_loss_D_B_R = self.backward_D_basic(self.s2_netD_B, self.s2_real_R, s2_fake_A_R)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.s1_idt_A_L = self.s1_netG_A(self.s1_sim_L)
            self.s1_loss_idt_A_L = self.criterionIdt(self.s1_idt_A_L, self.s1_sim_L) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.s1_idt_B_L = self.s1_netG_B(self.s1_real_L)
            self.s1_loss_idt_B_L = self.criterionIdt(self.s1_idt_B_L, self.s1_real_L) * lambda_A * lambda_idt

            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.s1_idt_A_R = self.s1_netG_A(self.s1_sim_R)
            self.s1_loss_idt_A_R = self.criterionIdt(self.s1_idt_A_R, self.s1_sim_R) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.s1_idt_B_R = self.s1_netG_B(self.s1_real_R)
            self.s1_loss_idt_B_R = self.criterionIdt(self.s1_idt_B_R, self.s1_real_R) * lambda_A * lambda_idt
        else:
            self.s1_loss_idt_A_L = 0
            self.s1_loss_idt_B_L = 0
            self.s1_loss_idt_A_R = 0
            self.s1_loss_idt_B_R = 0

        # GAN loss D_A(G_A(A))
        self.s1_loss_G_A_L = self.criterionGAN(self.s1_netD_A(self.s1_fake_B_sim_L), True)
        # GAN loss D_B(G_B(B))
        self.s1_loss_G_B_L = self.criterionGAN(self.s1_netD_B(self.s1_fake_A_real_L), True)

        # GAN loss D_A(G_A(A))
        self.s1_loss_G_A_R = self.criterionGAN(self.s1_netD_A(self.s1_fake_B_sim_R), True)
        # GAN loss D_B(G_B(B))
        self.s1_loss_G_B_R = self.criterionGAN(self.s1_netD_B(self.s1_fake_A_real_R), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.s1_loss_cycle_A_L = self.criterionCycle(self.s1_rec_A_real_L, self.s1_real_L) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.s1_loss_cycle_B_L = self.criterionCycle(self.s1_rec_B_sim_L, self.s1_sim_L) * lambda_B

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.s1_loss_cycle_A_R = self.criterionCycle(self.s1_rec_A_real_R, self.s1_real_R) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.s1_loss_cycle_B_R = self.criterionCycle(self.s1_rec_B_sim_R, self.s1_sim_R) * lambda_B

        # combined loss and calculate gradients
        self.s1_loss_G = self.s1_loss_G_A_L + self.s1_loss_G_A_R + self.s1_loss_G_B_L + self.s1_loss_G_B_R + \
                        self.s1_loss_cycle_A_L + self.s1_loss_cycle_A_R + self.s1_loss_cycle_B_L + self.s1_loss_cycle_B_R + \
                        self.s1_loss_idt_A_L + self.s1_loss_idt_A_R + self.s1_loss_idt_B_L + self.s1_loss_idt_B_R

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.s2_idt_A_L = self.s2_netG_A(self.s2_sim_L)
            self.s2_loss_idt_A_L = self.criterionIdt(self.s2_idt_A_L, self.s2_sim_L) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.s2_idt_B_L = self.s2_netG_B(self.s2_real_L)
            self.s2_loss_idt_B_L = self.criterionIdt(self.s2_idt_B_L, self.s2_real_L) * lambda_A * lambda_idt

            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.s2_idt_A_R = self.s2_netG_A(self.s2_sim_R)
            self.s2_loss_idt_A_R = self.criterionIdt(self.s2_idt_A_R, self.s2_sim_R) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.s2_idt_B_R = self.s2_netG_B(self.s2_real_R)
            self.s2_loss_idt_B_R = self.criterionIdt(self.s2_idt_B_R, self.s2_real_R) * lambda_A * lambda_idt
        else:
            self.s2_loss_idt_A_L = 0
            self.s2_loss_idt_B_L = 0
            self.s2_loss_idt_A_R = 0
            self.s2_loss_idt_B_R = 0

        # GAN loss D_A(G_A(A))
        self.s2_loss_G_A_L = self.criterionGAN(self.s2_netD_A(self.s2_fake_B_sim_L), True)
        # GAN loss D_B(G_B(B))
        self.s2_loss_G_B_L = self.criterionGAN(self.s2_netD_B(self.s2_fake_A_real_L), True)

        # GAN loss D_A(G_A(A))
        self.s2_loss_G_A_R = self.criterionGAN(self.s2_netD_A(self.s2_fake_B_sim_R), True)
        # GAN loss D_B(G_B(B))
        self.s2_loss_G_B_R = self.criterionGAN(self.s2_netD_B(self.s2_fake_A_real_R), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.s2_loss_cycle_A_L = self.criterionCycle(self.s2_rec_A_real_L, self.s2_real_L) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.s2_loss_cycle_B_L = self.criterionCycle(self.s2_rec_B_sim_L, self.s2_sim_L) * lambda_B

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.s2_loss_cycle_A_R = self.criterionCycle(self.s2_rec_A_real_R, self.s2_real_R) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.s2_loss_cycle_B_R = self.criterionCycle(self.s2_rec_B_sim_R, self.s2_sim_R) * lambda_B

        # combined loss and calculate gradients
        self.s2_loss_G = self.s2_loss_G_A_L + self.s2_loss_G_A_R + self.s2_loss_G_B_L + self.s2_loss_G_B_R + \
                        self.s2_loss_cycle_A_L + self.s2_loss_cycle_A_R + self.s2_loss_cycle_B_L + self.s2_loss_cycle_B_R + \
                        self.s2_loss_idt_A_L + self.s2_loss_idt_A_R + self.s2_loss_idt_B_L + self.s2_loss_idt_B_R


        # depth loss
        self.dep_loss = self.model_loss(self.cs_outputs, self.real_gt, self.mask, dlossw=[float(e) for e in self.opt.dlossw.split(",") if e])

        self.loss_G = (self.s1_loss_G + self.s2_loss_G) * 0.5 + self.dep_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.s1_optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.s2_optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_cascade.zero_grad()
        self.backward_G()             # calculate gradients for G_A and G_B
        self.s1_optimizer_G.step()       # update G_A and G_B's weights
        self.s2_optimizer_G.step()       # update G_A and G_B's weights
        self.optimizer_cascade.step()
        # D_A and D_B
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.s1_optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.s2_optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.s1_optimizer_D.step()  # update D_A and D_B's weights
        self.s2_optimizer_D.step()  # update D_A and D_B's weights
