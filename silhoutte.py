from PIL import Image
import sys
import numpy as np
import imageio
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt
import matplotlib.pyplot as plt


"""
Crop to the desired input size for our model (28,28)
If the image has more than one channel (e.g. 4 in the case of screenshotted pngs), then convert to grayscale
"""


def test(filename, to_filename):

    ifile = imageio.imread(filename)
    ofile = imageio.imread(to_filename)

    print("Input size {} Output size {}".format(ifile.shape, ofile.shape))

def visualize(edt_img, cdt_img):
    # edt_img = (edt_img ** 2) 
    # edt_img = edt_img / np.max(edt_img)
    # edt_img = edt_img // 1
    # cdt_img = (cdt_img ** 2)
    # cdt_img = cdt_img / np.max(cdt_img)
    # cdt_img = cdt_img // 1
    # edt = Image.fromarray(edt_img, 'L')
    # cdt = Image.fromarray(cdt_img, 'L')
    # edt.show()
    # cdt.show()
    # img.save('my.png')
    # img.show()
    print ('just look at output image for now')


def get_heatmap_from_greyscale(img, scale):
    """
    :param img: a PIL image object
    :return: the heatmap output from distance_transform_edt
    """
    arr = np.array(img)
    print('shape', arr.shape)

    shadow = np.where(arr < 255, 0, 1)
    edt_dist_matrix = distance_transform_edt(shadow)
    edt_dist_matrix = edt_dist_matrix**scale
    #edt_dist_matrix /= np.max(np.abs(edt_dist_matrix), axis=0) #normalization

    # plt.imshow(edt_dist_matrix, cmap="gray")
    # plt.show()


    return Image.fromarray(edt_dist_matrix)



def shadow_distance(to_filename):
    '''
    finds outermost nonwhite
    pixels and crops out white pixels
    outside of the bounds
    returns cropped Pillow Image

    **** note: initial image must have enough whitespace around
    it to create the white margins. Else the pixel values which
    strech beyond the bounds of the original image are automatically
    filled in as black
    '''
    img = imageio.imread(to_filename) #read in as np array
    shadow = np.where(img < 255, 0, 1) # not white
    img = imageio.imsave(to_filename, img)
    edt_dist_matrix = distance_transform_edt(shadow) // 1
    cdt_dist_matrix = distance_transform_cdt(shadow) // 1 # need ints
    # edt_dist_matrix = utils.softmax(edt_dist_matrix)
    # cdt_dist_matrix = utilssoftmax(cdt_dist_matrix)
    imageio.imsave(to_filename[:-4] + 'edt' + to_filename[-4:], edt_dist_matrix)
    imageio.imsave(to_filename[:-4] + 'cdt' + to_filename[-4:], cdt_dist_matrix)
    return edt_dist_matrix, cdt_dist_matrix

def eliminate_whitespace(to_filename, image_buffer=5):
    '''
    finds outermost nonwhite
    pixels and crops out white pixels
    outside of the bounds
    returns cropped Pillow Image

    **** note: initial image must have enough whitespace around
    it to create the white margins. Else the pixel values which
    strech beyond the bounds of the original image are automatically
    filled in as black
    '''
    img = imageio.imread(to_filename) #read in as np array
    nonwhite_pixels = np.argwhere(img < 255) # not white
    num_cols = nonwhite_pixels.shape[0]
    
    upper = nonwhite_pixels[0,0]
    lower = nonwhite_pixels[num_cols-1,0]
    left = np.min(nonwhite_pixels[np.arange(num_cols),1])
    right = np.max(nonwhite_pixels[np.arange(num_cols),1])
    height = lower - upper
    width = right - left

    assert (height > 0)
    assert (width > 0)

    # ## use this to visualize where cropping is being done
    # ## at different stages by moving code around
    # img[upper,:] = 0
    # img[lower,:] = 0
    # img[:,left] = 0
    # img[:,right] = 0
    # imageio.imsave(to_filename, img)
    # input()
    # ##

    vertical_dim_bigger = height > width
    horizontal_dim_bigger = width > height
    if vertical_dim_bigger:
        margin = (height - width) // 2
        left -= margin
        right += margin
    elif horizontal_dim_bigger:
        margin = (width - height) // 2
        upper -= margin
        lower += margin
    left -= image_buffer
    right += image_buffer
    upper -= image_buffer
    lower += image_buffer
    box = (left, upper, right, lower)
    img = Image.open(to_filename)
    img = img.crop(box)
    return img

def main():

    print(sys.argv)
    filename = sys.argv[1]
    to_filename = sys.argv[2]
    # image_buffer = None
    # if len(sys.argv) == 4:
    #     image_buffer = sys.argv[3]
    img = Image.open(filename).convert('L')
    img.save(to_filename)

    edt_img, cdt_img = shadow_distance(to_filename)
    visualize(edt_img, cdt_img)
    # if image_buffer is not None:
    #     img = eliminate_whitespace(to_filename, int(image_buffer))
    # else:
    #     img = eliminate_whitespace(to_filename)
    # img.save(to_filename)
    # img = img.resize((28,28))
    # img.save(to_filename)


    # # Read back in as a numpy array
    # img = imageio.imread(to_filename)
    # img = 255 - img
    # imageio.imsave(to_filename, img)

    # test(filename, to_filename)


if __name__=="__main__":
    main()