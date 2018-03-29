from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#training parameters
num_steps = 100000 #original was 100.000
batch_size = 128
learning_rate = 0.0002

#network parameters
image_dim = 784 #28x28 pixel
num_hidden_gen = 256
num_hidden_disc = 256
num_noise = 100

#custom initialization Xavier Glorot
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0]/2.))

#store layer weight & bias
weights = {
    "gen_hidden_1": tf.Variable(glorot_init([num_noise, num_hidden_gen])),
    "gen_out": tf.Variable(glorot_init([num_hidden_gen, image_dim])),
    "disc_hidden_1": tf.Variable(glorot_init([image_dim, num_hidden_disc])),
    "disc_out": tf.Variable(glorot_init([num_hidden_disc, 1]))
}

biases = {
    "gen_hidden_1": tf.Variable(tf.zeros([num_hidden_gen])),
    "gen_out": tf.Variable(tf.zeros([image_dim])),
    "disc_hidden_1": tf.Variable(tf.zeros([num_hidden_disc])),
    "disc_out": tf.Variable(tf.zeros([1]))
}

#Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights["gen_hidden_1"])
    hidden_layer = tf.add(hidden_layer, biases["gen_hidden_1"])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights["gen_out"])
    out_layer = tf.add(out_layer, biases["gen_out"])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

#Discriminator
def discriminator(x):
    hidden_layer = tf.matmul(x, weights["disc_hidden_1"])
    hidden_layer = tf.add(hidden_layer, biases["disc_hidden_1"])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights["disc_out"])
    out_layer = tf.add(out_layer, biases["disc_out"])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

#Build Network
#Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, num_noise], name="input_noise")
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name="input_disc")

#Build Generator Network
gen_sample = generator(gen_input)

#Build 2 discriminators (1 from noise input 1 from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

#Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

#Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

#Training variables for each optimizer
#Tensorflow updates every variable of each optimizer,
#we need to store the variables to tell tensorflow which one to update

#Generator network variables
gen_vars = [weights["gen_hidden_1"], weights["gen_out"],
            biases["gen_hidden_1"], biases["gen_out"]]
#Discriminator network variables
disc_vars = [weights["disc_hidden_1"], weights["disc_out"],
             biases["disc_hidden_1"], biases["disc_out"]]

#Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

#Assign all variables
init = tf.global_variables_initializer()

#Start training
with tf.Session() as sess:

    #Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):
        #Prepare data
        #Get next batch of MNIST data (only images, no Labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        #Generate noise to feed the generator
        z = np.random.uniform(-1., 1., size=[batch_size, num_noise])

        #Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print("Losses(%i): Generator: %f, Discriminator:%f" % (i, gl, dl))

    #Generate images from noise, using the generator network
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        #Noise input
        z = np.random.uniform(-1., 1., size=[4, num_noise])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        #Reverse colors for better display
        g = -1 * (g-1)
        for j in range(4):
            #Generate image from noise, extend to 3 channels for matplot figure
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()