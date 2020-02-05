import pandas as pd
import pickle
import random
import gensim
import pickle
import os
from nltk.tokenize import word_tokenize
import re
import numpy as np


def create_image_csv():
    rootdir = '/images/CUB_200_2011/CUB_200_2011/images'
    df = pd.DataFrame(columns=['images'])
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
                df.loc[len(df)] = [file]
    df.to_csv("intermediate_results.csv", index=False)


def create_final_csv():
    rootdir = '/captions/text_c10'
    images = pd.read_csv('intermediate_results.csv')
    images['captions'] = 'abc'
    print(images.head())
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".txt"):
                with open(filepath) as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
                t = file.replace('.txt', '.jpg')
                images.loc[images['images'] == t, 'captions'] = content[random.randint(0, 1)]
    images.to_csv("final.csv", index=False)


def clean_and_tokenize_comments_for_image(comment):
    stop_words = ['a', 'and', 'of', 'to']
    punctuation = r"""!"#$%&'()*+,./:;<=>?@[\]^_`…’{|}~"""
    comments_without_punctuation = [s.translate(str.maketrans(' ', ' ', punctuation)) for s in comment]
    sentences = []

    for q_w_c in comments_without_punctuation:
        q_w_c = re.sub(r"-(?:(?<!\b[0-9]{4}-)|(?![0-9]{2}(?:[0-9]{2})?\b))", ' ', q_w_c)  # replace with space

        temp_tokens = word_tokenize(str(q_w_c).lower())
        tokens = [t for t in temp_tokens if t not in stop_words]
        sentences.append(tokens)
    return sentences


def create_feature_vectors_for_single_comment(word2vec_model, cleaned_comments, image_names):
    vectorized_list = []
    image_list = []

    for comments, image in zip(cleaned_comments, image_names):
        result_array = np.empty((0, 300))
        for word in comments:
            try:
                w = [word2vec_model[word]]
                result_array = np.append(result_array, w, axis=0)
            except KeyError:
                if word in 'superciliary' or word in 'superciliaries':
                    result_array = np.append(result_array, [word2vec_model['eyebrow']], axis=0)
                    result_array = np.append(result_array, [word2vec_model['region']], axis=0)
                elif word in 'rectrices' or word in 'rectices':
                    result_array = np.append(result_array, [word2vec_model['large']], axis=0)
                    result_array = np.append(result_array, [word2vec_model['tail']], axis=0)
                    result_array = np.append(result_array, [word2vec_model['feathers']], axis=0)
                else:
                    print(word)
                    result_array = np.append(result_array, [word2vec_model[random.choice(word2vec_model.index2entity)]], axis=0)

        vectorized_list.append(np.mean(result_array, axis=0).astype('float32'))
        image_list.append(image)

    return image_list, np.array(vectorized_list)


def create_sentence_embeddings():
    df = pd.read_csv('final.csv')
    model = gensim.models.KeyedVectors.load_word2vec_format('/word2vec_pretrained_model/GoogleNews-vectors-negative300.bin', binary=True)
    cleaned_captions = clean_and_tokenize_comments_for_image(df['captions'].values)
    image_names = df['images'].values
    print('Done tokenizing....')
    i, c = create_feature_vectors_for_single_comment(model, cleaned_captions, image_names)
    word_vector_dict = dict(zip(i, c))
    pickle.dump(word_vector_dict, open('word_vector_min_bird' + ".p", "wb"))
    print('Done')


if __name__ == '__main__':
    # create_image_csv() --> run only when intermediate_results.csv does not exist
    # create_final_csv() --> run only when final.csv does not exist
    create_sentence_embeddings()
