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

from custom import ClassificationModel



# constant type for image, heatmap, and label tensors
my_dtype = torch.float16

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

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
    :return: None
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

        imageio.imwrite(image_name[:-4] + '_r' + image_name[-4:], res)

    else:

        print('option B')

        image = image[:, :-image.shape[1] // 26, :]  # strip off the z

        res = np.zeros((5 * 64, 5 * 64, 3))
        for i in range(5):
            lbound = 64 * 5 * i
            rbound = 64 * 5 * (i + 1)
            res[i * 64:(i + 1) * 64, :] = image[:, lbound:rbound]

        imageio.imwrite(image_name[:-4] + '_r' + image_name[-4:], res)




"""

def load_alphabet(loader, image_name):
    
    # Split the alphabet image given by filename into the 26 characters and return them
    # :param filename: the path to an image
    # :return:
    

    image = Image.open(image_name)
    width, height = image.size
    images = []

    for i in range(0, width, width // 26):

        img = image.crop(i, 0, i + width // 26, height)
        img = loader(img).unsqueeze(0)
        img.append(img.to(device, torch.float))

    return images

"""



def image_loader(image_name, reshape=False, get_heatmap=False):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # implementation of image_loader from a pytorch tutorial
    # all this has since been migrated to utils under the name ut_image_loader, so that 
    # we can also handle getting image data for classification

    use_name = image_name

    if reshape:
        im_reshape(image_name)
        use_name = image_name[:-4] + '_r' + image_name[-4:] # match the reshaped filename


    image = Image.open(use_name)

    # Get the heatmap for the content image
    if get_heatmap:

        # TODO: for heatmap, convert directly from ndarray to tensor, then unsqueeze and send to device

        greyscale = image.convert("L")
        heatmap = loader(get_heatmap_from_greyscale(greyscale)).unsqueeze(0).to(device, torch.float)
    else:
        heatmap = None


    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float), heatmap
    






def ut_image_loader(image_name, loader, device, reshape=False, get_heatmap=False, classification=False):
    """
    params:
    image_name (string): the image that we're loading
    loader (torch transform): transformations to apply to the image
    device (device):
    reshape (bool): if we're doing style transfer, then there will be reshaped files with _r added
                    to the filename which we should look for
    get_heatmap (bool): if we want the heatmap for the distance loss, then set this flag so the heatmap
                    will be computed as well
    classification (bool): if we're loading these images for classification, then rather than returning a 
                    (320,320) image, we'll return 26 smaller images representing the individual characters,
                    and their corresponding labels
    """

    imh, imw = params["target_img_height"], params["target_img_width"]

    if classification:

        image = Image.open(image_name)
        image = loader(image)

        print(image.size, image.mode)

        # at this point, image has dimension (imh, (imw*26))

        # split the image into 26 distinct characters
        #images = image.view(3, imh, imh, 26)
        #images = torch.transpose(images, 0, 2) # swap character and height dims
        #images = torch.transpose(images, 1, 2) # swap height and width dimensions

        # should now have images of shape (26, imh, imw) 
        # where dim[0] = characters, dim[1] = width, dim[2] = height
        labels = torch.from_numpy(np.arange(26))    

        return image.to(device, my_dtype), labels.type(my_dtype)


    else:

        use_name = image_name

        if reshape:
            im_reshape(image_name)
            use_name = image_name[:-4] + '_r' + image_name[-4:] # match the reshaped filename


        image = Image.open(use_name)

        # Get the heatmap for the content image
        if get_heatmap:

            # TODO: for heatmap, convert directly from ndarray to tensor, then unsqueeze and send to device

            greyscale = image.convert("L")
            heatmap = loader(get_heatmap_from_greyscale(greyscale)).unsqueeze(0).to(device, my_dtype)
        else:
            heatmap = None


        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)

        return image.to(device, my_dtype), heatmap




def create_special_font_dataset(device, path, mode):
    """
    Run this a single time to create the classification dataset
    """
    fullpath = path + mode + "/"
    files = [f for f in listdir(fullpath) if isfile(join(fullpath, f))]


    imh, imw = params["target_img_height"], params["target_img_width"]


    # some redundancy here from what was in run.py
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    
    # 3, 64, 1664
    # Unless one of us wants to dig into creating a custom transformation, 
    # if we use color images (which I think we should), we'll have to res
    loader = transforms.Compose([
        #transforms.Resize((64//2, 1664//2)),  # scale imported image to half size
        transforms.ToTensor()])  # transform it into a torch tensor

    # assume we're always working with color images
    data_x = torch.zeros((0, 3, imh, imw)).type(my_dtype)
    data_y = torch.zeros((0,)).type(my_dtype)

    for file in files:

        # (26, 32, 32), (26,)
        imgs, labels = ut_image_loader(fullpath + file, loader, device, classification=True)

        # add the new images and labels to the full dataset
        data_x = torch.cat((data_x, imgs), 0)
        data_y = torch.cat((data_y, labels), 0)

    # now save the data to file so that we don't have to do this again
    print("saving under names:", mode + "_data_x.pt", mode + "_data_y.pt")
    torch.save(data_x, mode + "_data_x.pt")
    torch.save(data_y, mode + "_data_y.pt")






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

    return train_data_loader, val_data_loader




def visualize_samples(loader):

	examples = enumerate(loader)
	batch_idx, (example_data, example_targets) = next(examples)

	fig = plt.figure()
	for i in range(6):
		plt.subplot(2,3,i+1)
		plt.tight_layout()
		plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
		plt.title('Ground truth: {}'.format(example_targets[i]))
		#print(example_data[i][0])
		plt.xticks([])
		plt.yticks([])
	plt.show()






    



    
