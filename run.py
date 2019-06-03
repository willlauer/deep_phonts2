
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

from solver import Solver
from tqdm import tqdm

from hyper_params import params

import numpy as np

from custom import * #ContentLoss, StyleLoss, Normalization, DistanceTransform, Classification
from utils import * #style_layers_default, im_reshape

from silhoutte import get_heatmap_from_greyscale

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               prim_style_img, sec_style_img, device, heatmap,
                               style_layers=style_layers_default, use_classification_loss=False):

    """
    :param cnn:
    :param normalization_mean:
    :param normalization_std:
    :param prim_style_img:
    :param sec_style_img:
    :param device:
    :param heatmap: should be a tensor as in the paper

    :param content_layers:
    :param style_layers:
    :return:
    """

    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    prim_style_losses = []
    sec_style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():

        if i == 0:
            """
            Only add this for the first layer
            """
            USE_DISTANCE = True

            if USE_DISTANCE:
                x_content = model(sec_style_img).detach()
                distance_loss = DistanceTransform(x_content, heatmap)
                model.add_module("distance_loss", distance_loss)
            else:
                distance_loss = None

        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add primary style loss:
            target_feature = model(prim_style_img).detach()
            prim_style_loss = StyleLoss(target_feature)
            model.add_module("prim_style_loss_{}".format(i), prim_style_loss)
            prim_style_losses.append(prim_style_loss)

            # add secondary style loss:
            target = model(sec_style_img).detach()
            sec_style_loss = StyleLoss(target)
            model.add_module("sec_style_loss_{}".format(i), sec_style_loss)
            sec_style_losses.append(sec_style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, prim_style_losses, sec_style_losses, distance_loss

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def load_or_train_classifier(model_name):
    
    """
    Searches for a particular model name. If it exists, then load the saved weights and return the model.
    If it doesn't exist, then create a new model, train it, and then return it
    """

    # (num_classes, in_channel, c1, c2, c3)
    print("Load or train classifier")

    model = SmallVGG(25, 3, 8, 8, 5)

    training_batch_size, val_batch_size = params['batch_size_train'], params['batch_size_val']

    model_already_trained = False
    path = "./saved_models/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for f in tqdm(files):
        if f.find(model_name) != -1:
            model_already_trained = True 
            break
            
    # load the saved model
    if model_already_trained:
        model.load_state_dict(torch.load("./saved_models/" + model_name))
        return model

    # otherwise we have to train it 
    else:
        print("Training the model")
        train_loader, val_loader = get_classification_data_loader(training_batch_size, val_batch_size)

        solver = Solver(model, train_loader, val_loader)
        solver.train(params["train_num_epochs"])

        # after training, save the model
        torch.save(model.state_dict(), path + "/" + model_name)

    return model

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       prim_style_img, sec_style_img, input_img, device, heatmap, num_steps=300,
                       prim_style_weight=1000, sec_style_weight=1000, distance_weight=2000):
    """
    Run the style transfer
    """
    print('Building the style transfer model..')

    model, prim_style_losses, sec_style_losses, distance_losses = get_style_model_and_losses(cnn,
    normalization_mean, normalization_std, prim_style_img, sec_style_img, device, heatmap)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]

    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            prim_style_score = 0
            sec_style_score = 0

            for sl in prim_style_losses:
                prim_style_score += sl.loss
            for cl in sec_style_losses:
                sec_style_score += cl.loss

            prim_style_score *= prim_style_weight
            sec_style_score *= sec_style_weight

            loss = 0
            if distance_losses is not None:
                distance_score = distance_weight * distance_losses.loss
                loss = prim_style_score + sec_style_score + distance_score
            else:
                loss = prim_style_score + sec_style_score

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))

                if distance_losses is not None:
                    print('Primary Style Loss : {:4f} Secondary Style Loss: {:4f} Distance Loss: {:4f}'.format(
                        prim_style_score.item(), sec_style_score.item(), distance_score.item()))
                else:
                    print('Primary Style Loss : {:4f} Secondary Style Loss: {:4f}'.format(
                        prim_style_score.item(), sec_style_score.item()))
                print()

            if distance_losses is not None:
                return prim_style_score + sec_style_score + distance_score
            else:
                return prim_style_score + sec_style_score


        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    # prim_style_img, _ = image_loader("./data/images/Capitals_colorGrad64/test/ARACNE-CONDENSED_regular_italic.0.2.png")
    prim_style_img, _ = image_loader("./data/images/blue_texture.jpg")

    sec_style_img, heatmap = image_loader("./data/images/Capitals_colorGrad64/test/keyrialt.0.2.png",
                                        get_heatmap=True)

    # ensure style and content image are the same size
    assert prim_style_img.size() == sec_style_img.size(), \
        "we need to import style and content images of the same size"

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    plt.ion()

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # print content and style images
    plt.figure()
    imshow(prim_style_img, title='Primary Style Image')
    plt.figure()
    imshow(sec_style_img, title='Secondary Style Image')

    # import the model from pytorch pretrained models
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # get the classification model
    
    # KEEP THESE PYLINT COMMENTS! Not necessary, but vscode was giving a false-positive "too-many-arguments"
    # error, which the pylint comments disables locally
    # pylint: disable=E1121
    print("starting model creation")
    classification = load_or_train_classifier("classification_model.pt")
    print("ending model creation")
    # pylint: enable=E1121

    # vgg networks are trained on images with each channel normalized by mean [0.485, 0.456, 0.406] and
    # standard deviation [0.229, 0.224, 0.225]. Normalize the image using these values before sending it
    # to the network
    cnn_normalization_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device)
    cnn_normalization_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device)

    USE_RANDOM_NOISE = False
    if not USE_RANDOM_NOISE:
        input_img = sec_style_img.clone()
    else:   # if you want to use white noise instead uncomment the below line:
        input_img = torch.randn(sec_style_img.data.size(), device=device)

    # add the original input image to the figure:
    plt.figure()
    imshow(input_img, title='Input Image')

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                prim_style_img, sec_style_img, input_img, device, heatmap, num_steps=100)

    plt.figure()
    imshow(output, title='Output Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()