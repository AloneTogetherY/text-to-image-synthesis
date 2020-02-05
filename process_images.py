import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import pickle
from matplotlib import pyplot as plt
from keras.preprocessing.image import array_to_img

import pandas as pd
import os


def load_images():
    image_names = list()
    data_list = list()

    rootdir = '/images/CUB_200_2011/CUB_200_2011/images'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
                print(file)
                pixels = load_img(filepath, target_size=(64, 64)) # resize all images to 64x64
                # convert to numpy array
                pixels = img_to_array(pixels)
                # store
                image_names.append(file)
                data_list.append(pixels)

    return image_names, np.asarray(data_list)


def save_preprocessed_image_set(n_name='image_names.npy', a_name='image_arrays.npz'):
    image_names, image_array = load_images()

    np.save(n_name, image_names)

    print('Done saving images')

    np.savez_compressed(a_name, image_array)

    print("Done saving preprocessed data")


def load_and_normalize_images(images='image_arrays.npz', image_names='image_names.npy', img_vectors='image_vectors'):
    imgs = np.load(images)['arr_0']
    img_names = np.load(image_names)

    image_name_image_vectors_dict = dict(zip(img_names, imgs))

    # Normalize image vectors from 0-255 to [-1, 1] and save results
    for k in image_name_image_vectors_dict:
        image_name_image_vectors_dict[k] = (image_name_image_vectors_dict[k].astype('float32') - 127.5) / 127.5

    pickle.dump(image_name_image_vectors_dict, open(img_vectors + ".p", "wb"))

    print("done")


def display_random_images():
    images = np.load('image_arrays.npz')['arr_0']
    image_names = np.load('image_names.npy')
    ix = np.random.randint(0, len(images), 50)
    images_t = images[ix]
    image_names = image_names[ix]
    n = 3
    for i in range(n * n):
        print(i, image_names[i])
        df = pd.read_csv('final.csv', ',', None)
        print(df.loc[df['images'] == image_names[i], 'captions'].item())
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(array_to_img(images_t[i]))
    plt.show()


if __name__ == '__main__':
    save_preprocessed_image_set()
    load_and_normalize_images()
