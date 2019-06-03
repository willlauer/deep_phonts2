import torch
from torch import nn
import torch.nn.functional as F
from models.SmallVGG import SmallVGG
from utils import *
from tqdm import tqdm
import imageio
from torch.autograd import Variable
from torch.distributions.normal import Normal
from hyper_params import params



class Solver:

    def __init__(self, model, train_loader, val_loader):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.normal = Normal(0, 1)

    def check_accuracy(self):
        """
        Print out the accuracy across the entire validation set, and the accuracy over
        ten minibatches of the training dataset
        :return: None
        """
        self.model.eval()
        with torch.no_grad():

            num_correct = 0
            num_samples = 0

            for x, y in self.val_loader:
                y = y.type(torch.long)
                scores, _, _ = self.model.forward(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

            acc = float(num_correct) / num_samples
            print('Validation Set: Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))

        with torch.no_grad():

            count = 0
            num_correct = 0
            num_samples = 0

            for x, y in self.train_loader:
                y = y.type(torch.long)
                if count == 6: # do this for 6 batches
                    break
                count += 1
                scores, _, _ = self.model.forward(x)
                _, preds =  scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)


            acc = float(num_correct) / num_samples
            print('Training Set: Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))

        # Set the model back into training mode
        self.model.train()

    def transfer(self, num_iters, filename_a, filename_b):

        """
        Assuming we have a pre-trained model from the result of train(), perform style transfer
        from some image a to image b

        :return: None
        """

        # pull out the letters used and remove the file extension
        name_a = filename_a[filename_a.rfind('/')+1:][:-4]
        name_b = filename_a[filename_b.rfind('/')+1:][:-4]




        content_img = torch.from_numpy(imageio.imread(filename_a)).float()
        content_img = content_img.expand(1, 1, content_img.shape[0], content_img.shape[1])
        style_img = torch.from_numpy(imageio.imread(filename_b)).float()
        style_img = style_img.expand(1, 1, style_img.shape[0], style_img.shape[1])

        self.model.mode = 'transfer'    # we now want to compute content and style

        # Compute the content and style of our content and style images
        _, content_target, _ = self.model.forward(content_img)
        _, _, style_target = self.model.forward(style_img)

        # sample from normal distribution, wrap in a variable, and let requires_grad=True
        # torch._C.Variable used to prevent compiler error when accessing data attribute
        noise = torch._C.Variable(self.normal.sample(content_target.shape), requires_grad=True)
        print(content_target.shape)

        optimizer = torch.optim.Adam([noise], lr=params["transfer_lr"])

        store_every = 10000
        for i in tqdm(range(num_iters)):

            if (i+1) % store_every == 0:
                imageio.imwrite('transfer_checkpoint_images/{}_to_{}_{}.jpg'.format(name_a, name_b, i), noise.data.squeeze())

            # send our noise forward through the model, computing its style and content
            optimizer.zero_grad()
            _, content, style = self.model.forward(noise)

            # compute the loss as the sum of the mean-squared-error loss for the content and style
            content_loss = F.mse_loss(content, content_target)
            style_loss = sum([F.mse_loss(style[i], style_target[i]) for i in range(len(style))])



            loss = params["content_weight"] * content_loss \
                   + params["style_weight"] * style_loss

            if (i+1) % store_every == 0:
                print(content_loss, style_loss)

            # compute gradient with respect to the input and take a step
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


    def train(self, num_epochs):

        optimizer = torch.optim.Adam(self.model.parameters())
        for ep in range(num_epochs):

            ct = 0
            print('Epoch {}'.format(ep))
            for x,y in self.train_loader:

                # to keep F.cross_entropy happy
                y = y.type(torch.long)

                optimizer.zero_grad()
                scores, _, _ = self.model.forward(x)
                loss = F.cross_entropy(scores, y)

                optimizer.zero_grad()

                loss.backward() # compute gradients
                optimizer.step()

                ct += 1

                if ct % params["print_every"] == 0:
                    print('Iteration {}, loss {}'.format(ct, loss.item()))
                    self.check_accuracy()
