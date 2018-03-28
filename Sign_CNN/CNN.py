#Import Data
from __future__ import division, print_function, absolute_import
from data import train_images, train_labels
from data import test_images, test_labels
import tensorflow as tf
import random
import matplotlib.pyplot as plt

#training parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 100

#network parameters
num_input = 784 #28x28 images
num_classes = 62 #62 classes of signs
dropout = 0.25 #25% dropout for training

#create the neural network
def conv_net(x_dict, n_classes, dropout,reuse, is_training):
    #define a scope for reusing the variables
    with tf.variable_scope("ConvNet", reuse=reuse):
        #inpute layer
        x = x_dict["x"]
        #pictures are 28x28 in size and 784 in 1D
        #[batch_size, height, width,channel]
        x = tf.reshape(x, [-1, 28, 28, 1])

        #Convolutional Layer with 32 filters and a kernel size of 5
        conv_1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        #Max-Pooling the layer with kernel size 2 and stride of 2
        conv_1 = tf.layers.max_pooling2d(conv_1, 2, 2)

        #convolution layer with 64 filters and a kernel size of 5
        conv_2 = tf.layers.conv2d(conv_1, 64, 5, activation=tf.nn.relu)
        #Max-Pooling the layer with kernel size 2 and stride of 2
        conv_2 = tf.layers.max_pooling2d(conv_2, 2, 2)

        #fully connected layer by flatten conv_2-layer
        fully_connected_1 = tf.contrib.layers.flatten(conv_2)

        #fully connected layer (from contrib to layer)
        fully_connected_1 = tf.layers.dense(fully_connected_1, 1024)
        #apply the dropout if the model is training ( is_Training=True)
        fully_connected_1 = tf.layers.dropout(fully_connected_1, rate=dropout, training=is_training)

        #output layer class prediction
        out = tf.layers.dense(fully_connected_1, num_classes)
    return out

#define th model function ( tf estimator template

def model_fn(features, labels, mode):
    #built the neural network
    #2 neural networks needed, because dropout has different behaviour, but needs the same weights
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True) #no reuse for training one
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    #prediction
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    #if prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    #Define Loss and Optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)
    ))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    #Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    #tf estimators require to return a EstimatorSpec, that specify
    #the different ops for training, evaluation, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={"accuracy": acc_op})
    return estim_specs

#build the estimator
model = tf.estimator.Estimator(model_fn)

#define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_images}, y=train_labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
#train the model
model.train(input_fn, steps=num_steps)


#evaluate the model
#define the input function for evaluation
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_images}, y=test_labels,
batch_size=batch_size, shuffle=False)
#use the estimator evaluate method
accuracy_score = model.evaluate(input_fn)["accuracy"]
print("Accuracy:", accuracy_score)

#pick 10 random pictures and compare them
sample_indexes = random.sample(range(len(test_images)), 10)
sample_images = [test_images[i] for i in sample_indexes]
sample_labels = [test_labels[i] for i in sample_indexes]

#display prediction in plot
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": sample_images[i]},num_epochs=1, shuffle=False    )
    predictions = list(model.predict(input_fn=input_fn))
    prediction = predictions[0]
    plt.subplot(5, 2, 1+i)
    plt.axis("off")
    color = "green" if truth == prediction else "red"
    plt.text(40, 10, "Truth:      {0}\nPrediction: {1}".format(truth,prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")
plt.show()