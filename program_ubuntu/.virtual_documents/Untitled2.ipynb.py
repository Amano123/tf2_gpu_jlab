get_ipython().getoutput("pip install tensorflow_datasets pandas")


import tensorflow as tf
import tensorflow_datasets as tfds
import os
import io
import glob
import numpy as np
import re


def preprocess_sentence(w, num=None):
    w = '<start> ' + w + ' <end>'
    if num == None:
        return w
    else:
        return w, num


sample_text = '【sample text】'
print(preprocess_sentence(sample_text))


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


def create_dataset(file_path, file_names):
    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    TAKE_SIZE = 20000
    labeled_data_sets = []
    for i, file_name in enumerate(file_names, 0):
        lines_dataset = tf.data.TextLineDataset(os.path.join(file_path, file_name)).map(preprocess_sentence)
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    all_labeled_data = all_labeled_data.shuffle(
                                        BUFFER_SIZE, reshuffle_each_iteration=False
                                        )
    return all_labeled_data


files_path = "./sudachi_dataset/"
DATA_FILE_NAMES = ['バイク.txt', '公開企業.txt']
all_labeled_data = create_dataset(file_path=files_path, file_names=DATA_FILE_NAMES)
for text, label in all_labeled_data:
    print(text.numpy().decode("utf-8"))
    print(label)
    break



