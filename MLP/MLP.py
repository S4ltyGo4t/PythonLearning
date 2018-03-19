import numpy as np
import tensorflow as tf
import random

#generate test data
#list of 100 random numbers from 1 - 100
data = []
data = np.random.randint(100, size=100)

"""
   Input     Hidden1           Hidden2            Output

              h1.1              h2.1 
  x1(1)       h1.2              h2.2        
              h1.3              h2.3              y1(3)
  x2(2)       h1.4              h2.4
              h1.5              h2.5

"""
#parameters
num_input = 2
num_output = 1
num_hidden1 = 5
num_hidden2 = 5
learning_rate = 0.01
batch_size = 5
epochs = 200

#2 Inputs(x) 1 Output(y)
x = tf.placeholder(tf.float32, [None, num_input], name="X")
y = tf.placeholder(tf.float32, [None, num_output], name="Y")

#weights
w1 = tf.Variable(tf.random_normal([num_input, num_hidden1]))
w2 = tf.Variable(tf.random_normal([num_hidden1, num_hidden2]))
wout = tf.Variable(tf.random_normal([num_hidden2, num_output]))
#biases
b1 = tf.Variable(tf.random_normal([num_hidden1]))
b2 = tf.Variable(tf.random_normal([num_hidden2]))
bout = tf.Variable(tf.random_normal([num_output]))

#build model
layer_1 = tf.add(tf.matmul(x, w1), b1)
layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
logits = tf.nn.relu(tf.add(tf.matmul(layer_2, wout), bout))

#loss function and optimizer
loss_op = tf.losses.mean_squared_error(y, logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(loss_op)

#init viriables
init = tf.global_variables_initializer()
accuracy = tf.reduce_mean(tf.square(tf.subtract(y, logits)))

#run training
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(data)/batch_size)

    #results before training the MLP
    test_x = []
    test_x.append(random.choice(data))
    test_x.append(random.choice(data))
    test_x = np.reshape(test_x, (1, 2))
    test_y = sum(test_x)
    test_y = np.reshape(test_y, (-1, 1))
    before = sess.run(loss_op, feed_dict={x: test_x, y: test_y})

    #Training
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            #get 2 random numbers for input from data list
            batch_x = []
            batch_x.append(random.choice(data))
            batch_x.append(random.choice(data))
            batch_x = np.reshape(batch_x, (1, 2))
            #calculate the expected result
            batch_y = sum(batch_x)
            batch_y = np.reshape(batch_y, (-1, 1))
            #run the training session
            _, c = sess.run([optimizer, loss_op], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c/total_batch
            if epoch % 10 == 0:
                #print progress
                print('Epoch:', (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost))

    #Testing, results after training
    #inpute data
    test_x = []
    test_x.append(random.choice(data))
    test_x.append(random.choice(data))
    test_x = np.reshape(test_x, (1, 2))
    #correct output
    test_y = sum(test_x)
    test_y = np.reshape(test_y, (-1, 1))
    #feed the MLP
    after = sess.run(loss_op, feed_dict={x: test_x, y: test_y})
    print("Before: ", before, " After: ", after)