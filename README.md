# deep_phonts2
Living and learning. This time with pre-trained weights and more interesting data!

Todo:

- [ ] experiment with different layers in content loss

Baseline layer [conv(1-5)]

Test 1: Remove layers [conv(1,3,5)] 

Test 2: Remove layers and go deeper [conv(1,3,5,7,9)]

Test 3: Extremely sparse layer sampling [conv(1,5,9)]

Will (1,2,3)

Test 5: Only sample early layers [conv(1,2,3)]

Nick

Test 6: Ignore early layers [conv(4,5,6)]

Nick

Test 7: Sparsity starting at deeper layers [conv(4, 6, 8, 10)]

Nick

Store results like layer_test_[convlayers].png. e.g. layer_test46810.png


- [ ] experiment with weights

Vary weighting for primary style loss, secondary style loss, primary distance loss, secondary distance loss. Possibly use the classifier to modify these weights at runtime.


- [ ] defuck the style

After the final report, before the poster session.

- [ ] resolve noise in letters

Some letters aren't fully visible, have missing bits, etc. Might be easier to do in post-processing. Outline of the character is supposed to be handled by distance loss, the fill of the character is handled by the style loss.


- [ ] single character trials


images (directory): containing all of the image data that we use as inputs to the model

outputs (directory): contains all of the style transfer outputs, in form _style_[alphabet1]_content_[alphabet2].png 

# Data is from:
https://people.eecs.berkeley.edu/~sazadi/MCGAN/datasets/
We're only using Capitals_colorGrad64. Download it, move it into the main deep_phonts2 directory, and run 
tar -zxvf Capitals_colorGrad64.tar.gz -C ./data/images
