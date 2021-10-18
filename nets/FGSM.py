import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import uniform
from torch.autograd import Variable

def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)

        dist = dist.view(x.shape[0], -1)

        dist_norm = torch.norm(dist, dim=1, keepdim=True)

        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)

        # dist = F.normalize(dist, p=2, dim=1)

        dist = dist / dist_norm

        dist *= epsilon

        dist = dist.view(x.shape)

        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x

class FastGradientSignUntargeted():
    b"""
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, epsilon=0.2, alpha=0.025, min_val=-2, max_val=2, max_iters=1, _type='linf'):
        self.model = model

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type

    def perturb(self, imgL, imgR, img_L_transformed, img_R_transformed, disp_gt, input_L_warped, input_R_warped, mask, reduction4loss='mean', isTrain=True):
        # original_images: values are within self.min_val and self.max_val

        change = torch.abs(input_R_warped - imgR)
        change.clamp_(0.01, 2)
        # The adversaries created from random close points to the original data
        distribution = uniform.Uniform(-change, change)
        rand_perturbR = distribution.sample()
        rand_perturbR = rand_perturbR.cuda()

        change = torch.abs(input_L_warped - imgL)
        change.clamp_(0.01, 2)
        # The adversaries created from random close points to the original data
        distribution = uniform.Uniform(-change, change)
        rand_perturbL = distribution.sample()
        rand_perturbL = rand_perturbL.cuda()

        xL = Variable(imgL.data, requires_grad=True).cuda()
        xL = xL + rand_perturbL
        xL.clamp_(self.min_val, self.max_val)
        xR = Variable(imgR.data, requires_grad=True).cuda()
        xR = xR + rand_perturbR
        xR.clamp_(self.min_val, self.max_val)

        xLT = Variable(img_L_transformed.data, requires_grad=True).cuda()
        xLT = xLT + rand_perturbL
        xLT.clamp_(self.min_val, self.max_val)
        xRT = Variable(img_R_transformed.data, requires_grad=True).cuda()
        xRT = xRT + rand_perturbR
        xRT.clamp_(self.min_val, self.max_val)

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                loss = 0
                if isTrain:
                    pred_disp1, pred_disp2, pred_disp3 = self.model(xL, xR, xLT, xRT)
                    pred_disp = pred_disp3
                    loss = 0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt[mask], reduction='mean') \
                           + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt[mask], reduction='mean') \
                           + F.smooth_l1_loss(pred_disp3[mask], disp_gt[mask], reduction='mean')
                else:
                    pred_disp = self.model(xL, xR, xLT, xRT)
                    loss = F.smooth_l1_loss(pred_disp[mask], disp_gt[mask], reduction='mean')

                if reduction4loss == 'none':
                    grad_outputs = torch.ones(loss.shape).cuda()

                else:
                    grad_outputs = None


                xL = self.aug(xL, imgL, loss, grad_outputs)
                xR = self.aug(xR, imgR, loss, grad_outputs)
                xLT = self.aug(xLT, img_L_transformed, loss, grad_outputs)
                xRT = self.aug(xRT, img_R_transformed, loss, grad_outputs)

                self.model.zero_grad()
        img_L_transformed.data = xLT.data
        img_R_transformed.data = xRT.data
        return xL, xR, img_L_transformed, img_R_transformed

    def aug(self, x, img, loss,grad_outputs):
        grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
                only_inputs=True, retain_graph=True)[0]
        x.data += self.alpha * torch.sign(grads.data)
        x = project(x,img,self.epsilon, self._type)
        x.clamp_(self.min_val, self.max_val)
        x = Variable(x.data, requires_grad=True)
        return x
