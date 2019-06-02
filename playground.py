import torch
import torchvision.transforms as transforms
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

data = np.array([[0.1, 0.1, 0.8], [0.2, 0.3, 0.5], [0.05, 0.1, 0.85]])
data = torch.from_numpy(data)

y = np.array([2, 2, 2])
y = torch.from_numpy(y)

print(F.cross_entropy(data, y))




"""
im = imageio.imread('./data/images/Capitals_colorGrad64/train/Sintony-Regular.0.1_resized.png')

print(im.shape)


im = torch.from_numpy(im)

#im2 = im[:64, :64, :]
#im2 = im.view(25, 64, 64, 3).numpy()[0,:,:,:]
#im2 = np.transpose(im2[0,:,:,:]), (1, 2, 0))

imgs = []
for i in range(5):
    for j in range(5):
        imgs.append(im[i*64:(i+1)*64, j*64:(j+1)*64,:])


im2 = imgs[5]

fig = plt.figure()

plt.subplot(2,1,1)
plt.tight_layout()
plt.imshow(im, interpolation='none')
plt.xticks([])
plt.yticks([])

plt.subplot(2,1,2)
plt.tight_layout()
plt.imshow(im2, interpolation='none')
plt.xticks([])
plt.yticks([])


plt.show()


"""


"""
im = Image.open("./data/images/Capitals_colorGrad64/train/8blimro.0.1.png")

print(im.size)
print(im.mode)

#im = im.resize(28, 28)

print(im.size)

ops = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

#Try showing this to make sure the colors don't get messed up!
#plt.figure()
#    imshow(style_img, title='Style Image')
#
#    plt.figure()
#    imshow(content_img, title='Content Image')

im = ops(im)

print(im.shape)
"""