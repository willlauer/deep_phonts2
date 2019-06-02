
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
<<<<<<< HEAD
from tqdm import tqdm

from hyper_params import params
=======
>>>>>>> refs/remotes/origin/master

import numpy as np

from custom import * #ContentLoss, StyleLoss, Normalization, DistanceTransform, Classification
from utils import * #content_layers_default, style_layers_default, im_reshape

from silhoutte import get_heatmap_from_greyscale

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, device, heatmap,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default, use_classification_loss=False):

    """

    :param cnn:
    :param normalization_mean:
    :param normalization_std:
    :param style_img:
    :param content_img:
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
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)


    i = 0  # increment every time we see a conv
    for layer in cnn.children():

        if i == 0:
            """
            Only add this for the first layer
            """
            USE_DISTANCE = False

            if USE_DISTANCE:
                x_content = model(content_img).detach()
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

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break



    model = model[:(i + 1)]




    return model, style_losses, content_losses, distance_loss





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
<<<<<<< HEAD
    print("Load or train classifier")

    model = SmallVGG(25, 3, 8, 8, 5)

    training_batch_size, val_batch_size = params['batch_size_train'], params['batch_size_val']
=======


    model = SmallVGG(26, 3, 8, 8, 5)

    training_batch_size, val_batch_size = 3000, 1000
>>>>>>> refs/remotes/origin/master

    model_already_trained = False
    path = "./saved_models/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
<<<<<<< HEAD
    for f in tqdm(files):
=======
    for f in files:
>>>>>>> refs/remotes/origin/master
        if f.find(model_name) != -1:
            model_already_trained = True 
            break
            
    # load the saved model
    if model_already_trained:
        model.load_state_dict(torch.load("./saved_models/" + model_name))
        return model

    # otherwise we have to train it 
    else:
<<<<<<< HEAD
        print("Training the model")
=======
>>>>>>> refs/remotes/origin/master
        train_loader, val_loader = get_classification_data_loader(training_batch_size, val_batch_size)

        solver = Solver(model, train_loader, val_loader)
        solver.train(params["train_num_epochs"])

        # after training, save the model
        torch.save(model.state_dict(), path + "/" + model_name)

    return model







def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, device, heatmap, num_steps=300,
                       style_weight=1000000, content_weight=1, distance_weight=1):
    """
    Run the style transfer
    """
    print('Building the style transfer model..')



    model, style_losses, content_losses, dl = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, device, heatmap)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]


    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = 0
            if dl is not None:
                distance_score = distance_weight * dl.loss
                loss = style_score + content_score + distance_score
            else:
                loss = style_score + content_score

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))

                if dl is not None:
                    print('Style Loss : {:4f} Content Loss: {:4f} Distance Loss: {:4f}'.format(
                        style_score.item(), content_score.item(), distance_score.item()))
                else:
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                print()

            if dl is not None:
                return style_score + content_score + distance_score
            else:
                return style_score + content_score


        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img








def main():

<<<<<<< HEAD
=======
    

>>>>>>> refs/remotes/origin/master
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu



    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor


    style_img, _ = image_loader("./data/images/Capitals_colorGrad64/train/8blimro.0.1.png")
    #style_img, _ = ut_image_loader("./data/images/Capitals_colorGrad64/train/8blimro.0.1.png",\
    #                                loader, device, reshape=True)

    content_img, heatmap = image_loader("./data/images/Capitals_colorGrad64/train/18thCtrKurStart.0.2.png",
                                        get_heatmap=True)
    #content_img, heatmap = ut_image_loader("./data/images/Capitals_colorGrad64/train/18thCtrKurStart.0.2.png", \
    #                                        loader, device, reshape=True, get_heatmap=True)



    # print(style_img.shape, content_img.shape)



    # Show some of the images from the dataset

    assert style_img.size() == content_img.size(), \
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

    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')






    # import the model from pytorch pretrained models
    cnn = models.vgg19(pretrained=True).features.to(device).eval()


    # get the classification model
    
    # KEEP THESE PYLINT COMMENTS! Not necessary, but vscode was giving a false-positive "too-many-arguments"
    # error, which the pylint comments disables locally
    # pylint: disable=E1121
    print("starting model creation")
<<<<<<< HEAD
    classification = load_or_train_classifier("classification_model.pt")
=======
    #classification = load_or_train_classifier("classification_model.pt")
>>>>>>> refs/remotes/origin/master
    print("ending model creation")
    # pylint: enable=E1121


    # vgg networks are trained on images with each channel normalized by mean [0.485, 0.456, 0.406] and
    # standard deviation [0.229, 0.224, 0.225]. Normalize the image using these values before sending it
    # to the network
    cnn_normalization_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device)
    cnn_normalization_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device)


    USE_RANDOM_NOISE = False
    if not USE_RANDOM_NOISE:
        input_img = content_img.clone()
    else:   # if you want to use white noise instead uncomment the below line:
        input_img = torch.randn(content_img.data.size(), device=device)



    # add the original input image to the figure:
    plt.figure()
    imshow(input_img, title='Input Image')


    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, device, heatmap)

    plt.figure()
    imshow(output, title='Output Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()



if __name__=="__main__":
    main()