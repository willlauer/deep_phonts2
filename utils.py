import imageio
import numpy as np

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']



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