"""
Author: Isabella Liu 8/13/21
Feature: Cycle GAN Model
Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
"""

import torch
import itertools
from .gan_networks import define_G, define_D, GANLoss
from utils.image_poll import ImagePool
from utils.test_util import load_from_dataparallel_model


class CycleGANModel:
    def __init__(self, lambdaA=10.0, lambdaB=10.0, lambda_identity=0.5, isTrain=True):
        """
        lambdaA: weight for cycle loss (A -> B -> A)
        lambdaB: weight for cycle loss (B -> A -> B)
        lambda_identity: use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight
            of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller
            than the weight of the reconstruction loss, please set lambda_identity = 0.1
        """
        self.lambdaA = lambdaA
        self.lambdaB = lambdaB
        self.lambda_identity = lambda_identity
        self.isTrain = isTrain

        # Define networks for both generators and discriminators
        self.netG_A = define_G(input_nc=1, output_nc=1, ngf=64, netG='unet_128', norm='instance')
        self.netG_B = define_G(input_nc=1, output_nc=1, ngf=64, netG='unet_128', norm='instance')

        if self.isTrain:
            self.netD_A = define_D(input_nc=1, ndf=64, netD='basic')
            self.netD_B = define_D(input_nc=1, ndf=64, netD='basic')
            # Create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(pool_size=50)
            self.fake_B_pool = ImagePool(pool_size=50)
            # Define loss functions
            self.criterionGAN = GANLoss(gan_mode='lsgan')
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_device(self, device):
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B, self.criterionGAN]:
            net = net.to(device)

    def set_distributed(self, is_distributed, local_rank):
        """Set distributed training"""
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B]:
            if is_distributed:
                net = torch.nn.parallel.DistributedDataParallel(
                    net, device_ids=[local_rank], output_device=local_rank
                )
            else:
                net = torch.nn.DataParallel(net)

    def load_model(self, file_name):
        G_A_dict = torch.load(file_name)['G_A']
        G_B_dict = torch.load(file_name)['G_B']
        self.netG_A.load_state_dict(G_A_dict)
        self.netG_B.load_state_dict(G_B_dict)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input):
        self.real_A_L = input['img_L']
        self.real_A_R = input['img_R']
        self.real_B = input['img_real']

    def forward(self):
        # fake_B = G_A(A)
        self.fake_B_L = self.netG_A(self.real_A_L)
        self.fake_B_R = self.netG_A(self.real_A_R)
        # recover_A = G_B(fake_B)
        self.rec_A_L = self.netG_B(self.fake_B_L)
        self.rec_A_R = self.netG_B(self.fake_B_R)
        # fake_A = G_B(B)
        self.fake_A = self.netG_B(self.real_B)
        # recover_B = G_A(fake_A)
        self.rec_B = self.netG_A(self.fake_A)

    def compute_loss_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        if self.lambda_identity > 0:
            # G_A should output identical result as input if real_B is fed: ||G_A(real_B) - real_B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.lambdaB * self.lambda_identity
            # G_B should output identical result as input if real_A(L and R) is fed: ||G_B(real_A) - real_A||
            self.idt_B_L = self.netG_B(self.real_A_L)
            self.idt_B_R = self.netG_B(self.real_A_R)
            self.loss_idt_B_L = self.criterionIdt(self.idt_B_L, self.real_A_L) * self.lambdaA * self.lambda_identity
            self.loss_idt_B_R = self.criterionIdt(self.idt_B_R, self.real_A_R) * self.lambdaA * self.lambda_identity
            self.loss_idt_B = (self.loss_idt_B_L + self.loss_idt_B_R) * 0.5
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # G_A loss = D_A(G_A(real_A))
        self.loss_G_A_L = self.criterionGAN(self.netD_A(self.fake_B_L), True)
        self.loss_G_A_R = self.criterionGAN(self.netD_A(self.fake_B_R), True)
        self.loss_G_A = (self.loss_G_A_L + self.loss_G_A_R) * 0.5
        # G_B loss = D_B(G_B(real_B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss ||G_B(fake_B) - real_A||
        self.loss_cycle_A_L = self.criterionCycle(self.rec_A_L, self.real_A_L) * self.lambdaA
        self.loss_cycle_A_R = self.criterionCycle(self.rec_A_R, self.real_A_R) * self.lambdaA
        self.loss_cycle_A = (self.loss_cycle_A_L + self.loss_cycle_A_R) * 0.5

        # Backward cycle loss ||G_A(fake_A) - real_B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambdaB

        # Combine losses and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B
        return self.loss_G

    def compute_loss_D_basic(self, netD, real, fake):
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
        return loss_D

    def compute_loss_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B_L = self.fake_B_pool.query(self.fake_B_L)
        fake_B_R = self.fake_B_pool.query(self.fake_B_R)
        self.loss_D_A_L = self.compute_loss_D_basic(self.netD_A, self.real_B, fake_B_L)
        self.loss_D_A_R = self.compute_loss_D_basic(self.netD_A, self.real_B, fake_B_R)
        self.loss_D_A = (self.loss_D_A_L + self.loss_D_A_R) * 0.5

    def compute_loss_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B_L = self.compute_loss_D_basic(self.netD_B, self.real_A_L, fake_A)
        self.loss_D_B_R = self.compute_loss_D_basic(self.netD_B, self.real_A_R, fake_A)
        self.loss_D_B = (self.loss_D_B_L + self.loss_D_B_R) * 0.5

    def update_D(self):
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.compute_loss_D_A()     # calculate gradients for D_A
        self.loss_D_A.backward()
        self.compute_loss_D_B()     # calculate gradients for D_B
        self.loss_D_B.backward()
        self.optimizer_D.step() # update Ds weights

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # Update Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)   # Ds require no gradient when optimizing Gs
        self.optimizer_G.zero_grad()    # set Gs' gradient to zero
        self.compute_loss_G()   # calculate gradient for G_A and G_B
        self.loss_G.backward()
        self.optimizer_G.step()     # update Gs weights
        # Update Ds
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.compute_loss_D_A()     # calculate gradients for D_A
        self.loss_D_A.backward()
        self.compute_loss_D_B()     # calculate gradients for D_B
        self.loss_D_B.backward()
        self.optimizer_D.step() # update Ds weights

    def train(self):
        """Make models train mode during train time"""
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B]:
            net.train()

    def eval(self):
        """Make models eval mode during test time"""
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B]:
            net.eval()



if __name__ == '__main__':
    h, w = 256, 512
    img_L = torch.rand(1, 1, h, w).cuda()
    img_R = torch.rand(1, 1, h, w).cuda()
    img_real = torch.rand(1, 1, h, w).cuda()
    input = {'img_L': img_L, 'img_R': img_R, 'img_real': img_real}

    cuda_device = torch.device("cuda:{}".format(0))
    cycleGAN = CycleGANModel()
    cycleGAN.set_device(cuda_device)
    cycleGAN.set_input(input)

    # cycleGAN.forward()
    #
    # fake_B_L = cycleGAN.fake_B_L
    # fake_A = cycleGAN.fake_A
    #
    # print(fake_B_L.shape)
    # print(fake_A.shape)

    cycleGAN.optimize_parameters()
