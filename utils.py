import imageio
import numpy as np
import torchvision.transforms as transforms



from os import listdir
from os.path import isfile, join

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






def load_alphabet(loader, image_name):
    """
    Split the alphabet image given by filename into the 26 characters and return them
    :param filename: the path to an image
    :return:
    """

    image = Image.open(image_name)
    width, height = image.size
    images = []

    for i in range(0, width, width // 26):

        img = image.crop(i, 0, i + width // 26, height)
        img = loader(img).unsqueeze(0)
        img.append(img.to(device, torch.float))

    return images



def ut_image_loader(image_name, loader, reshape=False, get_heatmap=False, classification=False):


    if classification:

        image = Image.open(image_name)
        image = loader(image)

        # at this point, image has dimension (32, (32*26))


        # split the image into 26 distinct characters
        images = image.image.view(32, 32, 26)
        images = torch.transpose(images, 0, 2) # swap character and height dims
        images = torch.transpose(images, 1, 2) # swap height and width dimensions

        # should now have images of shape (26, 32, 32) 
        # where dim[0] = characters, dim[1] = width, dim[2] = height
        labels = torch.tensor(list(range(26)))    

        return images.to(device, torch.float), labels


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
            heatmap = loader(get_heatmap_from_greyscale(greyscale)).unsqueeze(0).to(device, torch.float)
        else:
            heatmap = None


        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)

        return image.to(device, torch.float), heatmap




def create_special_font_dataset(path="./data/images/Capitals_colorGrad64/", mode="train"):
    """
    Run this a single time to create the classification dataset
    """
    fullpath = path + mode + "/"
    files = [f for f in listdir(fullpath) if isfile(join(fullpath, f))]


    # some redundancy here from what was in run.py
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    
    # 64, 1664
    loader = transforms.Compose([
        transforms.Resize(64//2, 1664//2),  # scale imported image to half size
        transforms.ToTensor()])  # transform it into a torch tensor


    data_x = torch.zeros((0, 32, 32))
    data_y = torch.zeros((0,))

    for file in files:

        # (26, 32, 32), (26,)
        imgs, labels = ut_image_loader(file, loader, classification=True)

        # add the new images and labels to the full dataset
        data_x = torch.concat(data_x, imgs, 0)
        data_y = torch.concat(data_y, labels, 0)

    # now save the data to file so that we don't have to do this again
    torch.save(data_x, "data_x.pt")
    torch.save(data_y, "data_y.pt")


def get_classification_dataset(training_batch_size):

    """
    Returns two TensorDatasets for the image data and the corresponding labels
    Each image should be 32x32, and the labels are scalars representing the index into the 
    alphabet (e.g. a=0, b=1, c=2, d=3, etc ... )
    """

    path = "./data/"
    files = [f for f in listdir(path) if isfile(join(path, f))]

    contains_classification_data = False
    for f in files:
        if f.find("data_x.pt") != -1:
            contains_classification_data = True 
    if not contains_classification_data:
        create_special_font_dataset()

    data_x = torch.load("./data/data_x.pt")
    data_y = torch.load("./data/data_y.pt")



    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data_x, data_y),
        batch_size=training_batch_size)




def load_or_train_classifier(loader, mypath):

    training_batch_size = 300

    model_name = "classification_model.pt"

    path = "./models/"
    files = [f for f in listdir(path) if isfile(join(path, f))]

    data_loader = get_classification_data_loader(training_batch_size)

    


    
