import os
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import random

def load_data(data_directory):
    #Create a list of all directory listed  in data_directory
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    #data and data labels
    labels = []
    images = []

    #load image from directory to labels
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        files_names = [os.path.join(label_directory, f)
                       for f in os.listdir(label_directory)
                       if f.endswith(".ppm")]
        for f in files_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return images, labels

def get_batch(list, batch_size):
    ret = []
    random_index = random.sample(range(len(list)), batch_size)
    for i in random_index:
        ret.append(np.array(list[i]).flatten())
    return np.array(ret)


#DIRECTORYS for Test- & Train-Data
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.join(ROOT_DIR, "Training")
test_dir = os.path.join(ROOT_DIR, "Testing")

#TEST-DATA 28x28 in Grayscale
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

#train data
train_images = [transform.resize(image, (28, 28)) for image in train_images]
train_images = np.array(train_images)
train_images = rgb2gray(np.array(train_images))
train_images = train_images.astype(np.float32)

train_labels = np.array(train_labels)

#test data
test_images = [transform.resize(image, (28, 28)) for image in test_images]
test_images = np.array(test_images)
test_images = rgb2gray(np.array(test_images))
test_images = test_images.astype(np.float32)

test_labels = np.array(test_labels)