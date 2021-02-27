import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import glob
import os
import csv


def get_train_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-3] == class_names
    return tf.argmax(one_hot)

def get_val_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    name = parts[-1].numpy().decode()
    one_hot = val_labels[name] == class_names
    return tf.argmax(one_hot)

def decode_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img/255.0
    return img

def process_train_path(file_path):
    label = get_train_label(file_path)
    img = decode_img(file_path)
    return img, label

get_val_label_wrapper = lambda x: tf.py_function(get_val_label, [x], tf.int64)

def process_val_path(file_path):
    label = get_val_label_wrapper(file_path)
    img = decode_img(file_path)
    return img, label

def channel_first_and_normalize(img, lbl, mean, std):    # over batches
    with tf.device('/device:GPU:0'):
        img = tf.transpose(img, perm=[0,3,1,2])
        img = img - mean
        img = img / std
    return img, lbl


def augment(img, lbl, seed):         # over batches
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    img = tf.image.stateless_random_saturation(img, 0.3, 3, seed)
    img = tf.image.stateless_random_hue(img, 0.08, new_seed)
    img = tf.image.stateless_random_contrast(img, 0.4, 1, seed)
    img = tf.image.stateless_random_brightness(img, 0.3, new_seed)
    img = tf.clip_by_value(img, 0, 1)
    return img, lbl

rng = tf.random.Generator.from_seed(123, alg='philox')
# A wrapper function for updating seeds
def augment_wrapper(x, y):
    seed = rng.make_seeds(2)[0]
    with tf.device('/device:GPU:0'):
        img, lbl = augment(x, y, seed)
    return img, lbl

def get_augmentation_layers():
    data_augmentation = tf.keras.Sequential([                       # over batches
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomHeight(0.5),
        layers.experimental.preprocessing.RandomWidth(0.5),
        layers.experimental.preprocessing.RandomTranslation(0.5,0.5),
        layers.experimental.preprocessing.RandomRotation(0.15),     # 0.12 * 2pi = 43.2 deg
        layers.experimental.preprocessing.Resizing(img_height, img_width)
    ])
    return data_augmentation