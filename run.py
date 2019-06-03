
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
import sys


def get_style_model_and_losses(cnn, classifier, normalization_mean, normalization_std,
                               prim_style_img, sec_style_img, device, prim_heatmap, sec_heatmap,
                               layer_names=style_layers_default, use_distance=False,
                               use_classification=False):

    """
    :param cnn:
    :param classifier:
    :param normalization_mean:
    :param normalization_std:
    :param prim_style_img:
    :param sec_style_img:
    :param device:
    :param prim_heatmap: should be a tensor as in the paper
    :param sec_heatmap: should be a tensor as in the paper

    :param content_layers:
    :param style_layers:
    :return:
    """
    print('get style model and losses')
    print('use_distance, use_classification', use_distance, use_classification)

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
            Only add these for the first layer
            """


            if use_distance:
                prim_content = model(prim_style_img).detach()
                sec_content = model(sec_style_img).detach()
                prim_distance_loss = DistanceTransform(prim_content, prim_heatmap)
                model.add_module("prim_distance_loss", prim_distance_loss)
                sec_distance_loss = DistanceTransform(sec_content, sec_heatmap)
                model.add_module("sec_distance_loss", sec_distance_loss)
            else:
                prim_distance_loss = None
                sec_distance_loss = None


            if use_classification:
                classification_loss = ClassificationLoss(classifier)
                model.add_module("classification_loss", classification_loss)
            else:
                classification_loss = None



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

        if name in layer_names:
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
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, prim_style_losses, sec_style_losses, prim_distance_loss, sec_distance_loss, classification_loss

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

def run_style_transfer(cnn, classifier, normalization_mean, normalization_std,
                       prim_style_img, sec_style_img, input_img, prim_heatmap, sec_heatmap, device, num_steps=300,
                       prim_style_weight=1000, sec_style_weight=1000, prim_dist_weight=2000, sec_dist_weight=2000,
                       classifier_weight=500, use_distance=False, use_classification=False):
    """
    Run the style transfer
    """
    print('Building the style transfer model..')
    print('use_distance, use_classification', use_distance, use_classification)


    model, prim_style_losses, sec_style_losses, prim_distance_losses, sec_distance_losses, classifier_loss = get_style_model_and_losses(cnn,
                    classifier, normalization_mean, normalization_std, prim_style_img, sec_style_img, device, prim_heatmap, sec_heatmap,
                    use_distance=use_distance, use_classification=use_classification)

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
            #if sec_distance_losses is not None:
            if use_distance:
                prim_distance_score = prim_dist_weight * prim_distance_losses.loss
                sec_distance_score = sec_dist_weight * sec_distance_losses.loss
                loss = prim_style_score + sec_style_score + prim_distance_score + sec_distance_score
            else:
                loss = prim_style_score + sec_style_score
            
            if use_classification:
                classifier_score = classifier_weight * classifier_loss.loss
                loss += classifier_score.squeeze()

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))

                if use_distance:
                    print('Primary Style Loss : {:4f} Secondary Style Loss: {:4f} Primary Distance Loss: {:4f} Secondary Distance Loss: {:4f}'.format(
                        prim_style_score.item(), sec_style_score.item(), prim_distance_score.item(), sec_distance_score.item()))

                else:
                    print('Primary Style Loss : {:4f} Secondary Style Loss: {:4f}'.format(
                        prim_style_score.item(), sec_style_score.item()))

                if classifier_loss is not None:
                    print('Classifier Loss: {:4f}'.format(classifier_score.item()))

                print()

            #if sec_distance_losses is not None:
            if use_distance:
                return prim_style_score + sec_style_score + prim_distance_score + sec_distance_score
            else:
                return prim_style_score + sec_style_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img



def visualize(a, b, c, d):
    """
    Display the four images in a single plot
    a, b, c, d => prim_style, sec_style, input_img, output_img
    """
    a, b, c, d = a.cpu().clone(), b.cpu().clone(), c.cpu().clone(), d.cpu().clone()
    a, b, c, d = a.squeeze(), b.squeeze(), c.squeeze(), d.squeeze()

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    titles = ["prim_style", "sec_style", "input_img", "output_img"]
    li = [unloader(a), unloader(b), unloader(c), unloader(d)]
    
    fig = plt.figure()

    for i in range(len(li)):
        s = plt.subplot(2, 2, (i+1))
        s.set_title(titles[i])
        plt.tight_layout()
        plt.imshow(li[i], interpolation='none')
        plt.xticks([])
        plt.yticks([])

    


def main():

    use_distance = '-d' in sys.argv
    use_classification = '-c' in sys.argv

    #print('use_distance, use_classification', use_distance, use_classification)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    prim_style_img, prim_heatmap = image_loader("./data/images/Capitals_colorGrad64/test/ARACNE-CONDENSED_regular_italic.0.2.png",
                                                get_heatmap=True)

    sec_style_img, sec_heatmap = image_loader("./data/images/Capitals_colorGrad64/test/keyrialt.0.2.png",
                                        get_heatmap=True)

    # ensure style and content image are the same size
    assert prim_style_img.size() == sec_style_img.size(), \
        "we need to import style and content images of the same size"


    # import the model from pytorch pretrained models
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # get the classification model
    
    # KEEP THESE PYLINT COMMENTS! Not necessary, but vscode was giving a false-positive "too-many-arguments"
    # error, which the pylint comments disables locally
    # pylint: disable=E1121
    print("starting model creation")
    classifier = load_or_train_classifier("classification_model.pt")
    print("ending model creation")
    # pylint: enable=E1121

    # vgg networks are trained on images with each channel normalized by mean [0.485, 0.456, 0.406] and
    # standard deviation [0.229, 0.224, 0.225]. Normalize the image using these values before sending it
    # to the network
    cnn_normalization_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device)
    cnn_normalization_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device)

    USE_RANDOM_NOISE = True
    if USE_RANDOM_NOISE:
        input_img = torch.randn(prim_style_img.data.size(), device=device)
        print("Using random image as base")
    else:
        input_img = prim_style_img.clone()
        print("Using primary style image as base")

    # add the original input image to the figure:
    #plt.figure()
    #imshow(input_img, title='Input Image')

    input_img_copy = input_img.clone() # since we modify the input image

    output = run_style_transfer(cnn, classifier, cnn_normalization_mean, cnn_normalization_std,
                                prim_style_img, sec_style_img, input_img, prim_heatmap, sec_heatmap, device,
                                use_distance=use_distance, use_classification=use_classification,
                                prim_style_weight=1000, 
                                sec_style_weight=1000, 
                                prim_dist_weight=2000, 
                                sec_dist_weight=2000)

    visualize(prim_style_img, sec_style_img, input_img_copy, output)

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()