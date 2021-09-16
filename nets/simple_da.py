"""
Author: Isabella Liu 8/23/21
Feature: A simple implementation of Domain Adaptation (DA) on feature space
"""

import torch
import itertools
import torch.nn.functional as F

from nets.psmnet_submodule import FeatureExtraction
from nets.gan_networks import NLayerDiscriminator, GANLoss
from nets.psmnet_wto_fe import PSMNet


class SimpleDA:
    def __init__(self, feature_channel=32):
        self.feature_channel = feature_channel
        self.feature_extractor = FeatureExtraction()
        self.net_D = NLayerDiscriminator(input_nc=self.feature_channel)
        self.criterion_D = GANLoss(gan_mode='lsgan')
        self.stereo_net = PSMNet(maxdisp=192)
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.optimizer_feature_extractor = torch.optim.Adam(self.feature_extractor.parameters(),
                                                            lr=0.001, betas=(0.9, 0.999))
        self.optimizer_stereo = torch.optim.Adam(self.stereo_net.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.training = False
        self.train()


    def set_input(self, input):
        """
        img_L, img_R, img_real: [bs, 1, H, W]
        """
        self.img_L = input['img_L']
        self.img_R = input['img_R']
        self.img_real = input['img_real']
        self.disp_gt = input['disp_gt']
        self.mask = (self.disp_gt < 192) * (self.disp_gt > 0)  # Note in training we do not exclude bg


    def set_device(self, device):
        for net in [self.feature_extractor, self.net_D, self.criterion_D, self.stereo_net]:
            net = net.to(device)

    def set_distributed(self, is_distributed, local_rank):
        """Set distributed training"""
        for net in [self.feature_extractor, self.net_D, self.stereo_net]:
            if is_distributed:
                net = torch.nn.parallel.DistributedDataParallel(
                    net, device_ids=[local_rank], output_device=local_rank
                )
            else:
                net = torch.nn.DataParallel(net)

    def load_model(self, file_name):
        models_dict = torch.load(file_name)
        feature_extractor_dict = models_dict['feature_extractor']
        D_dict = models_dict['D']
        stereo_dict = models_dict['Stereo']
        self.feature_extractor.load_state_dict(feature_extractor_dict)
        self.net_D.load_state_dict(D_dict)
        self.stereo_net.load_state_dict(stereo_dict)

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

    def train(self):
        """Make models train mode during train time"""
        self.training = True
        for net in [self.feature_extractor, self.net_D, self.criterion_D, self.stereo_net]:
            net.train()

    def eval(self):
        """Make models eval mode during test time"""
        self.training = False
        for net in [self.feature_extractor, self.net_D, self.criterion_D, self.stereo_net]:
            net.eval()

    def forward_feature_extractor(self):
        """
        img_L_feature, img_R_feature, img_real_feature: [bs, 32, H/4, W/4]
        """
        # Feature extractor
        self.img_L_feature = self.feature_extractor(self.img_L)
        self.img_R_feature = self.feature_extractor(self.img_R)
        self.img_real_feature = self.feature_extractor(self.img_real)

    def forward_stereo(self):
        if self.training:
            self.pred_disp1, self.pred_disp2, self.pred_disp3 = self.stereo_net(
                self.img_L_feature, self.img_R_feature)
            self.pred_disp = self.pred_disp3
        else:
            self.pred_disp = self.stereo_net(self.img_L_feature, self.img_R_feature)


    def forward_without_grad(self):
        with torch.no_grad():
            self.forward_feature_extractor()
            self.forward_stereo()
            # Loss on feature extractor
            pred_real = self.net_D(self.img_real_feature)
            loss_feature_extractor = self.criterion_D(pred_real, True)  # reverse
            self.loss_feature_extractor = loss_feature_extractor
            # Loss D on real images
            loss_D_real = self.criterion_D(pred_real, False)
            self.loss_D_real = loss_D_real
            # Loss D on fake images
            pred_fake_L = self.net_D(self.img_L_feature)
            pred_fake_R = self.net_D(self.img_R_feature)
            loss_D_fake_L = self.criterion_D(pred_fake_L, True)
            loss_D_fake_R = self.criterion_D(pred_fake_R, True)
            loss_D_fake = (loss_D_fake_L + loss_D_fake_R) * 0.5
            self.loss_D_fake = loss_D_fake
            self.loss_D = (loss_D_fake + loss_D_real) * 0.5
            # Loss on Stereo
            loss_stereo = F.smooth_l1_loss(self.pred_disp[self.mask], self.disp_gt[self.mask], reduction='mean')
            self.loss_stereo = loss_stereo

    def test(self):
        self.eval()
        with torch.no_grad():
            # When testing, there's no real image
            self.img_L_feature = self.feature_extractor(self.img_L)
            self.img_R_feature = self.feature_extractor(self.img_R)
            self.pred_disp = self.stereo_net(self.img_L_feature, self.img_R_feature)


    def compute_loss_D(self):
        # Real
        pred_real = self.net_D(self.img_real_feature)
        loss_D_real = self.criterion_D(pred_real, True)
        # Fake
        pred_fake_L = self.net_D(self.img_L_feature)
        loss_D_fake_L = self.criterion_D(pred_fake_L, False)
        pred_fake_R = self.net_D(self.img_R_feature)
        loss_D_fake_R = self.criterion_D(pred_fake_R, False)
        self.loss_D = (loss_D_real + 0.5 * (loss_D_fake_L + loss_D_fake_R)) * 0.5

    def optimize(self):
        # Forward input to feature extractor
        self.forward_feature_extractor()
        # Backward on feature extractor
        self.set_requires_grad([self.stereo_net, self.net_D], False)
        self.set_requires_grad([self.feature_extractor], True)
        self.optimizer_feature_extractor.zero_grad()
        pred_real = self.net_D(self.img_real_feature)
        loss_feature_extractor = self.criterion_D(pred_real, True)     # reverse
        loss_feature_extractor.backward()
        self.optimizer_feature_extractor.step()
        # Backward on discriminator
        self.set_requires_grad([self.feature_extractor, self.stereo_net], False)
        self.set_requires_grad([self.net_D], True)
        self.optimizer_D.zero_grad()
        pred_fake_L = self.net_D(self.img_L_feature.detach())
        loss_D_fake_L = self.criterion_D(pred_fake_L, True)     # TODO sim features are in target feature space
        pred_fake_R = self.net_D(self.img_R_feature.detach())
        loss_D_fake_R = self.criterion_D(pred_fake_R, True)
        pred_real = self.net_D(self.img_real_feature.detach())
        loss_D_real = self.criterion_D(pred_real, False)
        loss_D_fake = (loss_D_fake_L + loss_D_fake_R) * 0.5
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        self.optimizer_D.step()
        # Backward on stereo task
        self.set_requires_grad([self.net_D], False)
        self.set_requires_grad([self.feature_extractor, self.stereo_net], True)
        self.forward_feature_extractor()    # TODO feature extractor has been updated, so need to compute again
        self.forward_stereo()
        self.optimizer_feature_extractor.zero_grad()
        self.optimizer_stereo.zero_grad()
        if self.training:
            loss_stereo = 0.5 * F.smooth_l1_loss(self.pred_disp1[self.mask], self.disp_gt[self.mask], reduction='mean') \
               + 0.7 * F.smooth_l1_loss(self.pred_disp2[self.mask], self.disp_gt[self.mask], reduction='mean') \
               + F.smooth_l1_loss(self.pred_disp3[self.mask], self.disp_gt[self.mask], reduction='mean')
        else:
            loss_stereo = F.smooth_l1_loss(self.pred_disp[self.mask], self.disp_gt[self.mask], reduction='mean')
        loss_stereo.backward()
        self.optimizer_stereo.step()
        self.optimizer_feature_extractor.step()

        self.loss_D = loss_D
        self.loss_feature_extractor = loss_feature_extractor
        self.loss_D_real = loss_D_real
        self.loss_D_fake = loss_D_fake
        self.loss_stereo = loss_stereo


    def backward_old(self):
        self.optimizer.zero_grad()
        # Sim input, only backward on D (since Stereo task will backward on feature extractor later)
        self.set_requires_grad([self.feature_extractor], False)
        pred_fake_L = self.net_D(self.img_L_feature)
        loss_D_fake_L = self.criterion_D(pred_fake_L, True)     # TODO sim features are in target feature space
        pred_fake_R = self.net_D(self.img_R_feature)
        loss_D_fake_R = self.criterion_D(pred_fake_R, True)
        loss_D_fake = (loss_D_fake_L + loss_D_fake_R) * 0.5
        # loss_D_fake.backward(retain_graph=True)
        # Real input, backward on D and feature extractor
        self.set_requires_grad([self.feature_extractor], True)
        pred_real = self.net_D(self.img_real_feature)
        loss_D_real = self.criterion_D(pred_real, False)
        # loss_D_real.backward()
        self.optimizer.step()
        self.loss_D = (loss_D_real + 0.5 * (loss_D_fake_L + loss_D_fake_R)) * 0.5
        self.loss_D.backward()
        self.optimizer.step()



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    h, w = 256, 512
    img_L = torch.rand(1, 1, h, w).cuda()
    img_R = torch.rand(1, 1, h, w).cuda()
    img_real = torch.rand(1, 1, h, w).cuda()
    img_disp_gt = torch.rand(1, 1, h, w).cuda()
    input = {'img_L': img_L, 'img_R': img_R, 'img_real': img_real, 'disp_gt': img_disp_gt}

    cuda_device = torch.device("cuda:{}".format(0))
    simpleDA = SimpleDA()
    simpleDA.set_device(cuda_device)
    simpleDA.set_input(input)

    simpleDA.optimize()

    print(simpleDA.img_L_feature.shape)     # torch.Size([1, 32, 64, 128])
    print(simpleDA.img_R_feature.shape)     # torch.Size([1, 32, 64, 128])
    print(f'Loss_D: {simpleDA.loss_D} Loss_stereo: {simpleDA.loss_stereo}')

    # simpleDA.compute_loss_D()
    # print(simpleDA.loss_D)

