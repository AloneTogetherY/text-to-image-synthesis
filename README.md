# Exploring GAN-CLS and MS-GAN with a bird-dataset

I combine the GAN-CLS by Reed et al. [1] and the MS-GAN by Mao et al. [2] and experiment with the caltech bird-dataset.


<img src="https://github.com/Yoan-D/exploring-text-to-image-synthesis-with-conditional-GANs/blob/master/readme_images/gan-cls-ms-gan.png" width="550" align="center">

## Usage
1. Please refer to the READMEs in the folder images, captions, and word2vec_pretrained_model to obtain the necessary data. 
2. Run ```python process_images.py``` to resize and normalize the images and generate numpy arrays.
3. Run ```python process_captions.py``` to generate sentence embeddings for the captions.
4. Upload the generated images vectors, sentence vectors and pretrained word2vec model to a Google Drive account.
5. Load data into jupyter notebook in Google Colab.
6. Run code snippets in Google Colab.

### References
[1] Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text-to-image synthesis. In Proceedings of The 33rd International Conference on Machine Learning, 2016. <br />
[2] Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, and Ming-Hsuan Yang. Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis, IEEE Conference on Computer Vision and Pattern Recognition. 2019.
