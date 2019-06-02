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

from hyper_params import params



def gram_matrix(x):

    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)




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


class ClassificationLoss(nn.Module):

    def __init__(self, classification_model):

        super(ClassificationLoss, self).__init__()
        
        self.classification_model = classification_model
        
        for param in self.classification_model.parameters():
            # freeze the weights in the classification network
            param.require_grad = False

    
    def forward(self, x):
        
        # This should take in the full 5x5 grid of characters, and should run classfication on each,
        # computing the softmax cross-entropy loss. By the fixed nature of our data samples, we know 
        # the ground truth of what the character at each position is supposed to be. The total classification
        # loss of this sample x will be the sum of the differences between each confidence and 1
        #print('__Forward')
        dim = params['input_character_dim']
        self.loss = torch.zeros(1)

        for i in range(5):
            for j in range(5):

                # x_letter should be of dimension (batch_size, 3, input_character_dim, input_character_dim)
                x_letter = x[:, :, i * dim: (i+1) * dim, j * dim: (j+1) * dim]
                #print(i, j)
                # print('x shape', x.shape, x_letter.shape)

                y = (torch.ones(x.shape[0]) * (i * 5 + j)).type(torch.long)
                scores, _, _ = self.classification_model.forward(x_letter)
                self.loss += F.cross_entropy(scores, y)

        return x



class DistanceTransform(nn.Module):

    def __init__(self, x_content, heatmap):
        """

        :param target: content_image
        """
        super(DistanceTransform, self).__init__()

        self.x_content = x_content.detach()
        self.heatmap = heatmap.detach()
        self.x_content_times_heatmap = self.x_content * self.heatmap
        self.x_content_times_heatmap = self.x_content_times_heatmap.detach()

    def forward(self, x_gen):

        # x is the input image
        # compute silhouette of the original content image (binary)
        # then given input image x, compute heat


        self.loss = F.mse_loss(self.x_content_times_heatmap, x_gen * self.heatmap)
        #self.loss = torch.sum(1/2 * (self.target * self.heatmap - x * self.heatmap) ** 2)
        return x_gen




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
        self.mean = torch.from_numpy(np.array(mean)).view(-1, 1, 1).type(torch.float)
        self.std = torch.from_numpy(np.array(std)).view(-1, 1, 1).type(torch.float)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std