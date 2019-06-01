import torch
import torchvision.transforms as transforms
from PIL import Image

im = Image.open("./data/images/Capitals_colorGrad64/train/8blimro.0.1.png")

print(im.size)
print(im.mode)

#im = im.resize(28, 28)

print(im.size)

ops = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
"""
Try showing this to make sure the colors don't get messed up!
plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')
"""
im = ops(im)

print(im.shape)