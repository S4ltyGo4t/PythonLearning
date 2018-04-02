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


#weights
weights = {
    "hidden_1": tf.Variable(tf.random_normal([num_input, num_hidden1])),
    "hidden_2": tf.Variable(tf.random_normal([num_hidden1, num_hidden2])),
    "out": tf.Variable(tf.random_normal([num_hidden2, num_output]))
}
#biases
biases = {
    "hidden_1": tf.Variable(tf.zeros([num_hidden1])),
    "hidden_2": tf.Variable(tf.zeros([num_hidden2])),
    "out": tf.Variable(tf.zeros([num_output]))
}

#build model
def MLP(x):
    hidden_layer_1 = tf.matmul(x,weights["hidden_1"])
    hidden_layer_1 = tf.add(hidden_layer_1, biases["hidden_1"])
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)
    hidden_layer_2 = tf.matmul(hidden_layer_1, weights["hidden_2"])
    hidden_layer_2 = tf.add(hidden_layer_2, biases["hidden_2"])
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)
    out = tf.matmul(hidden_layer_2, weights["out"])
    out = tf.add(out, biases["out"])
    out = tf.nn.sigmoid(out)
    return out

#2 Inputs(x) 1 Output(y)
x = tf.placeholder(tf.float32, [None, num_input], name="X")
y = tf.placeholder(tf.float32, [None, num_output], name="Y")

#build network
mlp = MLP(x)

#build loss function
# loss_op = tf.losses.mean_squared_error(y, mlp)
loss_op = tf.reduce_mean(mlp)

#build optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

#create learning variables
vars = [weights["hidden_1"], weights["hidden_2"], weights["out"],
        biases["hidden_1"], biases["hidden_2"], biases["out"]]

#create training operator
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op, var_list=vars)

#init viriables
init = tf.global_variables_initializer()

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
    for epoch in range(1, epochs+1):
        avg_cost = 0
        for i in range(total_batch):
            #get 2 random numbers for input from data list
            batch_x = []
            num_1 = random.choice(data)
            num_2 = random.choice(data)
            batch_x.append(num_1)
            batch_x.append(num_2)
            batch_x = np.reshape(batch_x, (1, 2))
            #calculate the expected result
            batch_y = num_1 + num_2
            batch_y = np.reshape(batch_y, (-1, 1))
            #run the training session
            t, c = sess.run([train_op, loss_op], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c/total_batch
            if epoch % 10 == 0 or epoch == 1:
                #print progress
                print('Epoch:', (epoch), 'cost =', '{:.3f}'.format(avg_cost))

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
    after, x_value, y_value = sess.run([loss_op, x, y], feed_dict={x: test_x, y: test_y})
    print("Before: ", before, " After: ", after)
    print(x_value)
    print(y_value)