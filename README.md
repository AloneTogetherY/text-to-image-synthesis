# Exploring GAN-CLS and MS-GAN with a bird-dataset

I combine the GAN-CLS by Reed et al. [1] and the MS-GAN by Mao et al. [2] and experiment with the caltech bird-dataset.

<p align="center">

<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/gan-cls-ms-gan.png" width="550">
</p>

## Usage
1. Please refer to the READMEs in the folder images, captions, and word2vec_pretrained_model to obtain the necessary data. 
2. Run ```python process_images.py``` to resize and normalize the images and generate numpy arrays.
3. Run ```python process_captions.py``` to generate sentence embeddings for the captions.
4. Upload the generated images vectors, sentence vectors and pretrained word2vec model to a Google Drive account.
5. Import the jupyter notebook ```Text2Image_GAN_MS.ipynb``` in Google Colab and load the data.
6. Run code snippets in Google Colab.

## Results

I trained the GAN model for 960 epochs with the ADAM optimizer [3] for the discriminator and generator with a learning rate of 0.000035 and beta_1=0.5. Most of the synthesized images do depict plausible colors and shapes of birds and there does seem to be a lot of diversity; however, the GAN did have some minor mode collapse problems when generating images based on made up captions as seen below.

<p align="center">

<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/screenshots.png" width="650">
</p>

<p align="center">

<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/red_birds.png" width="550">

</p>

<p align="center">

<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/yellow_birds.png" width="550">

</p>


### Interpolating between sentence vectors
<p align="center">

<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/interpol_1.png" width="550">

</p>
<p align="center">

<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/interpol_2.png" width="550">

</p>
<p align="center">

<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/interpol_3.png" width="550">

</p>



### References
[1] Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text-to-image synthesis. In Proceedings of The 33rd International Conference on Machine Learning, 2016. <br />
[2] Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, and Ming-Hsuan Yang. Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis, IEEE Conference on Computer Vision and Pattern Recognition. 2019. <br />
[3] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014.
