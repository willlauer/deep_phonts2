from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np




def gram_matrix(x):

    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch._C.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)





# TODO: define the classification model! should be similar to smallvgg. Actually, probably
# could just use smallvgg

class ClassificationModel(nn.Module):

    # Run further forward passes through fc layers to do prediction
    # These layers should be pre-trained to recognize the characters, and not be
    # updated further

    def __init__(self):

        super(ClassificationModel, self).__init__()

    def forward(self, x):
        
        # The idea is that at any point, we should be able to correctly predict the character that
        # is being displayed
        pass






class ContentLoss(nn.Module):

    # Important! This is not a true pytorch loss function. If we want to define loss functions as
    # pytorch Loss functions, we have to create a pytorch autograd function to recompute/implement the
    # gradient manually in the backward method

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x




class DistanceTransform(nn.Module):

    def __init__(self, target, heatmap):
        """

        :param target: content_image
        """
        super(DistanceTransform, self).__init__()

        self.target = target.detach()
        self.heatmap = heatmap

    def forward(self, x):

        # x is the input image
        # compute silhouette of the original content image (binary)
        # then given input image x, compute heat


        self.loss = F.mse_loss(self.target * self.heatmap, x * self.heatmap)
        #self.loss = torch.sum(1/2 * (self.target * self.heatmap - x * self.heatmap) ** 2)
        return x




class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x






class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.from_numpy(np.array(mean)).view(-1, 1, 1)
        self.std = torch.from_numpy(np.array(std)).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std