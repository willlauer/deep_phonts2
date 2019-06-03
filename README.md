# deep_phonts2
Living and learning. This time with pre-trained weights and more interesting data!

Todo:

- [ ] experiment with different layers in content loss
- [ ] experiment with weights
- [ ] defuck the style
- [ ] 

images (directory): containing all of the image data that we use as inputs to the model

outputs (directory): contains all of the style transfer outputs, in form _style_[alphabet1]_content_[alphabet2].png 

# Data is from:
https://people.eecs.berkeley.edu/~sazadi/MCGAN/datasets/
We're only using Capitals_colorGrad64. Download it, move it into the main deep_phonts2 directory, and run 
tar -zxvf Capitals_colorGrad64.tar.gz -C ./data/images
