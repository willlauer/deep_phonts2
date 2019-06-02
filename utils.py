import imageio
import numpy as np
import torchvision.transforms as transforms

from os import listdir
from os.path import isfile, join
from models.SmallVGG import SmallVGG
import torch
from PIL import Image

import matplotlib.pyplot as plt

from torch import nn
from silhoutte import get_heatmap_from_greyscale
from hyper_params import params

import time


USE_DISTANCE = False
USE_CLASSIFICATION = False


# constant type for image, heatmap, and label tensors
my_dtype = torch.float

# desired depth layers to compute style/content losses :
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


resize_marker = '_resized'

view_every = 200

# This should be set to false for the first time running the model, so that it will be
# trained for classification on some of the training dataset
# After the first run, Classifier_is_trained should
# be set to True. We then load the model and freeze the weights. 
CLASSIFIER_IS_TRAINED = False 

def get_classification_layers(sizes):
    """
    Return a list of nn Modules that we can use to run classification
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1], bias=True))
    return layers
        



def im_reshape(image_name):
    """
    Reshape the image into a 5x5 grid of the first 25 characters in the alphabet
    :param image_name:
    :return: boolean (option A = False, option B = True)
    """

    image = imageio.imread(image_name)

    if len(image.shape) == 2:

        print('option A')

        image = image[:,:-(image.shape[1]//26)] # strip off the z

        res = np.zeros((5*64, 5*64))
        for i in range(5):
            lbound = 64 * 5 * i
            rbound = 64 * 5 * (i+1)
            res[i*64:(i+1)*64,:] = image[:,lbound:rbound]

        imageio.imwrite(image_name[:-4] + resize_marker + image_name[-4:], res)
        print('wrote to name', image_name[:-4] + resize_marker + image_name[-4:])

        return False

    else:

        print('option B')

        image = image[:, :-image.shape[1] // 26, :]  # strip off the z

        res = np.zeros((5 * 64, 5 * 64, 3))
        for i in range(5):
            lbound = 64 * 5 * i
            rbound = 64 * 5 * (i + 1)
            res[i * 64:(i + 1) * 64, :] = image[:, lbound:rbound]

        imageio.imwrite(image_name[:-4] + resize_marker + image_name[-4:], res)
        print('wrote to name', image_name[:-4] + resize_marker + image_name[-4:])

        return True



def image_loader(image_name, get_heatmap=False, classification=False):
    
    """
    Takes in the image name that we want to process
    Three optional params: reshape, get_heatmap, classification
    reshape (boolean): should always be true, since we'll always want to reshape images
                        to being square
    get_heatmap (boolean): if this is true, then compute the heatmap for the distance loss
                        metric and return it as well
    classification (boolean): if True, then return the range (0,24) as the second element
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3, 64, 1664
    # Unless one of us wants to dig into creating a custom transformation, 
    # if we use color images (which I think we should), we'll have to res
    loader = transforms.Compose([
        transforms.Resize(params["input_complete_dim"]),  # scale imported image to half size
        transforms.ToTensor()])  # transform it into a torch tensor

    use_name = image_name

    # Since we always want to be working with square data, this should always be true
    #assert reshape == True

    #if reshape:
    if use_name.find(resize_marker) == -1:
        # if this image has not yet been resized, then do so and update the use_name
        im_reshape(image_name)
        use_name = image_name[:-4] + resize_marker + image_name[-4:] # match the reshaped filename


    # Always convert to RGB
    image = Image.open(use_name).convert("RGB")


    # Get the heatmap for the content image
    if get_heatmap:

        # TODO: for heatmap, convert directly from ndarray to tensor, then unsqueeze and send to device

        greyscale = image.convert("L")
        heatmap = loader(get_heatmap_from_greyscale(greyscale)).unsqueeze(0).to(device, torch.float)
    else:
        heatmap = None

    """
    image = np.array(image)
    if heatmap is not None:
        heatmap = np.array(heatmap)
    """

    if isinstance(image, np.ndarray):
        print('post processing shapes for image, heatmap:', image.shape, 'n/a' if heatmap is None else heatmap.shape)
    else:
        print('post processing shapes for image, heatmap:', image.size, image.mode)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)

    print('post-loader shape', image.shape)

    if classification:
        return image.to(device, torch.float), torch.from_numpy(np.arange(25)).to(device, torch.int)
    else:
        return image.to(device, torch.float), heatmap







def create_special_font_dataset(device, path, mode):
    """
    Run this a single time to create the classification dataset
    """
    fullpath = path + mode + "/"
    files = [f for f in listdir(fullpath) if isfile(join(fullpath, f))]

    count = 0

    dim = params["input_character_dim"]

    # assume we're always working with color images
    data_x = torch.zeros((0, 3, params["input_complete_dim"], params["input_complete_dim"])).type(my_dtype)
    data_y = torch.zeros((0,)).type(torch.int)

    num_files = len(files)
    for i, file in enumerate(files[:1000]):
        print(i, '/', num_files)
        # (26, 3, 64, 64), (26,)
        # image_loader returns the square image (160,160,3)
        # we then need to reshape it to be the 25 characters that we care about
        imgs, labels = image_loader(fullpath + file, get_heatmap=False, classification=True)

        # split into the individual images
        #imgs = imgs.view(25, 3, params["input_character_dim"], params["input_character_dim"])

        imgs = imgs.squeeze()

        # manually split the image into the 25 characters that comprise it
        split_imgs = []
        for r in range(5):
            for c in range(5):
                split_imgs.append(imgs[:, r*dim:(r+1)*dim, c*dim:(c+1)*dim])

        # aggregate into a (25, 3, dim, dim) array
        print(len(split_imgs), split_imgs[0].shape)
        imgs = torch.stack(split_imgs) #np.array(split_imgs)
        print('imgs shape', imgs.shape)

        if i % view_every == 0:
            check_img = np.transpose(imgs[0,:,:,:].numpy(), (1, 2, 0))
            print('check image has shape', check_img.shape)
            
            imageio.imwrite('./check_{}.png'.format(i), check_img)


        print('loaded images have shape', imgs.shape)
        print(type(labels))
        print(labels)

        # add the new images and labels to the full dataset
        data_x = torch.cat((data_x, imgs), 0)
        data_y = torch.cat((data_y, labels), 0)

        print('cur_data_x shape', data_x.shape)
        print('cur_data_y shape', data_y.shape)


    # now save the data to file so that we don't have to do this again
    print("saving under names:", mode + "_data_x.pt", mode + "_data_y.pt")
    torch.save(data_x, './data/' + mode + "_data_x.pt")
    torch.save(data_y, './data/' + mode + "_data_y.pt")

    print('count = ', count)






def get_classification_data_loader(training_batch_size, val_batch_size):

    """
    Returns dataloaders corresponding to the train and validation datasets
    Each image should be 32x32, and the labels are scalars representing the index into the 
    alphabet (e.g. a=0, b=1, c=2, d=3, etc ... )
    """
    print("Getting the classification data loaders")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "./data/"
    files = [f for f in listdir(path) if isfile(join(path, f))]


    # search through the files stored in the data directory for the .pt file containing the 
    # image classification dataset 
    contains_classification_data = False
    for f in files:
        if f.find("train_data_x.pt") != -1:
            print("Found the training data")
            contains_classification_data = True 
    


    if not contains_classification_data:

        print("Couldn't find the training data")
        create_special_font_dataset(device, "./data/images/Capitals_colorGrad64/", "train")
        create_special_font_dataset(device, "./data/images/Capitals_colorGrad64/", "val")

    print("classification datasets created")

    train_data_x = torch.load("./data/train_data_x.pt")
    train_data_y = torch.load("./data/train_data_y.pt")

    val_data_x = torch.load("./data/val_data_x.pt")
    val_data_y = torch.load("./data/val_data_y.pt")

    print(train_data_x.shape, train_data_y.shape)

    print("train, val classification data successfully loaded")

    # Create the training and validation loaders using tensor datasets constructed from the 
    # image data and the corresponding labels
    train_data_loader = torch.utils.data.DataLoader(
                            torch.utils.data.TensorDataset(train_data_x, train_data_y),
                            batch_size=training_batch_size
                        )

    print("created train loader")

    val_data_loader = torch.utils.data.DataLoader(
                            torch.utils.data.TensorDataset(val_data_x, val_data_y),
                            batch_size=val_batch_size
                        )    

    print("created val loader")                

    visualize_samples(val_data_loader)

    return train_data_loader, val_data_loader

def visualize_samples(loader):

    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data[0].shape, example_targets[0].shape)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        example_data2 = np.transpose(example_data[i].numpy(), (1, 2, 0))
        plt.imshow(example_data2, interpolation='none')
        plt.title('Ground truth: {}'.format(example_targets[i]))
        #print(example_data[i][0])
        plt.xticks([])
        plt.yticks([])
    plt.show()


