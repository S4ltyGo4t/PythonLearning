from __future__ import division, print_function, absolute_import
from data import train_images, get_batch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#trianing parameters
learning_rate = 0.0002
num_steps = 8000 #original 100.000
batch_size = 100

#network parameters
image_dim = 784 #28x28
hidden_gen_dim = 256
hidden_disc_dim = 256
num_noise = 100

#custon initialization Xaver Glorot
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0]/2.))

#store weights & biases
weights = {
    "gen_hidden_1": tf.Variable(glorot_init([num_noise, hidden_gen_dim])),
    "gen_out": tf.Variable(glorot_init([hidden_gen_dim, image_dim])),
    "disc_hidden_1": tf.Variable(glorot_init([image_dim, hidden_disc_dim])),
    "disc_out": tf.Variable(glorot_init([hidden_disc_dim, 1]))
}

biases = {
    "gen_hidden_1": tf.Variable(tf.zeros([hidden_gen_dim])),
    "gen_out": tf.Variable(tf.zeros([image_dim])),
    "disc_hidden_1": tf.Variable(tf.zeros([hidden_disc_dim])),
    "disc_out": tf.Variable(tf.zeros([1]))
}

def generator(x):
    hidden_layer = tf.matmul(x, weights["gen_hidden_1"])
    hidden_layer = tf.add(hidden_layer, biases["gen_hidden_1"])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights["gen_out"])
    out_layer = tf.add(out_layer, biases["gen_out"])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def discriminator(x):
    hidden_layer = tf.matmul(x, weights["disc_hidden_1"])
    hidden_layer = tf.add(hidden_layer, biases["disc_hidden_1"])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights["disc_out"])
    out_layer = tf.add(out_layer, biases["disc_out"])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

#build neural network
#network inputs

gen_input = tf.placeholder(tf.float32, shape=[None, num_noise], name="gen_input")  #original
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name="disc_input") #original

#build neural network
gen_sample = generator(gen_input)

#build 2 discriminators 1 for real 1 for fake images
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

#build loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

#build optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

#train variables for each optimizers
#tensorflow updates every variable of each optimizer,
#we need to store the variables und tell explicit which one ot update

#generator network variables
gen_vars = [weights["gen_hidden_1"], weights["gen_out"],
            biases["gen_hidden_1"], biases["gen_out"]]
#discriminator network variables
disc_vars = [weights["disc_hidden_1"], weights["disc_out"],
             biases["disc_hidden_1"], biases["disc_out"]]

#create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

#init variables
init = tf.global_variables_initializer()



with tf.Session() as sess:
    #run the initializer
    sess.run(init)

    #learning
    for i in range(1, num_steps+1):
        #prepare data
        #get next batch of data, no lables
        batch_x = get_batch(train_images, batch_size)

        #generate noise to feed the generator
        z = np.random.uniform(-1., 1., size=[batch_size, num_noise])

        #train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print("Losses(%i): Generator %f, Discriminator %f" % (i, gl, dl))

    #generate images from noise, using the generator network
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        #Noise input
        z = np.random.uniform(-1., 1., size=[4, num_noise])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        # g = -1 * (g - 1) #reverse colors
        for j in range(4):
            # Generate image from noise, extend to 3 channels for matplot figure
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)
            a[j][i].axis("off")
    f.show()
    plt.draw()
    plt.waitforbuttonpress()