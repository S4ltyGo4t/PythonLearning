import tensorflow as tf
import os
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
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

    return images , labels

def display_4_rng(image_List):
    #vector of 4 random ints from 0 to list-lenght
    random_List = [np.random.randint(0,len(image_List)),
                   np.random.randint(0, len(image_List)),
                   np.random.randint(0, len(image_List)),
                   np.random.randint(0, len(image_List))]

    #adding the images to a plot
    for i in range(len(random_List)):
        plt.subplot(1, 4, i + 1)
        plt.axis("off")
        # set color map to gray, else it whould youre heat map
        plt.imshow(image_List[random_List[i]], cmap="gray")
        plt.subplots_adjust(wspace=0.5)
        print("shape: {0}, min: {1}, max: {2}".format(image_List[random_List[i]].shape,
                                                      image_List[random_List[i]].min(),
                                                      image_List[random_List[i]].max()))
    #print the plot
    plt.show()

def memory_of_list(list):
#prints the flags and  memory usage of a list
    tmp = np.array(list)
    print("---Flags---")
    print(tmp.flags)
    print("Itemsize: "+ str(tmp.itemsize))
    print("total used bytes: " + str(tmp.nbytes))

def show_label_image_grid():
    #print a grid of all lables with their images
    unique_labels = set(labels)

    #initialize the figure and counter
    plt.figure(figsize=(15, 15))
    i = 1

    for label in unique_labels:
        #pick first image of each label
        image = images[labels.index(label)]

        #define 64 subplots
        plt.subplot(8, 8, i)
        #dont include axis
        plt.axis("off")
        #add a title to each subplot
        plt.title("Label {0} ({1})".format(label,labels.count(label)))

        i += 1

        plt.imshow(image)
    plt.show()

#Directories
ROOT_PATH = "D:\Sources\PythonLearning\TensorflowPractice"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

#load images from source
images, labels = load_data(train_data_directory)
test_images, test_labels = load_data(test_data_directory)
#resize image to 28x28 and set them to grayscale
images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
images28 = rgb2gray(images28)

test_images28 = [transform.resize(image,(28,28))for image in test_images]
test_images28 = rgb2gray(np.array(test_images28))

# #display 4 random images to check (28x28)re- and greyscale
# display_4_rng(images28)

# #was used to get a feeling for images and their labels
# show_label_image_grid()

#neural network

#placeholders images
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
#placeholders labels
y = tf.placeholder(dtype=tf.int32, shape=[None])

#Flattern the input (28x28) -> (784)
images_flat = tf.contrib.layers.flatten(x)

#fully connected layer logits for linearization
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

#define a loss function, softmaxe to classify that one entry is only classified in one class
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                        logits=logits))

#define a optimizer to minimize the error and reach a global minimum
#depending on the chosen optimizer tune the parameters learningrate or momentum
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#convert logits to label index
correct_pred = tf.argmax(logits, 1)

#define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(301):
    print("EPOC", i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
    if i % 10 == 0:
        print("Loss: ", loss)
    print("DONE WITH EPOCH")

#pick 10 random pictures and compare them
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

#run the correct_preed operation for plot sample
predicted_sample = sess.run([correct_pred], feed_dict={x: sample_images})[0]

#runn full prediction for accuracy
predicted_full = sess.run([correct_pred], feed_dict={x: test_images28})[0]
#calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted_full)])
accuracy_full = match_count/len(test_labels)

#print the accuracy of the NN
print("Accuracy: {:.3f}".format(accuracy_full))

#display prediction in plot
fig = plt.figure(figsize=(10,10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted_sample[i]
    plt.subplot(5, 2, 1+i)
    plt.axis("off")
    color = "green" if truth == prediction else "red"
    plt.text(40, 10, "Truth:      {0}\n Prediction: {1}".format(truth,prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")
plt.show()

