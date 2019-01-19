# The MNIST data set has handwritten digits from 0-9 with 55,000 images for training and 10,000 images for testing
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from PIL import Image

# Import MNIST dataset and store the data into the variable mnist
# The dataset consists of 28x28 pixeld handwritten digits
# To represent the labels of the actual digit drawn, one-hot-encoding is used,
# representing a 1-dimensional vector holding binary values.
# The vector holding v = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] represents the digit 2
#                               ^ due to the value of 1 stored in bin 2.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST data", one_hot = True)       #  hot-encoded y-label

# The hand writing consists of 28x28 pixel images which are then transposed into a 1D-Array of 784 pixels.
# Each of the pixels represents an unsigned byte range from 0 to 255 which determines the grayscale of the pixel.
# A black pixel holds the value 255 where as a white pixel is given a value of 0. Upon 50 shades of gray.

# For each of the three subsets, The data holds 55,000 images for training, 5000 for validation,
# and the remaining 10,000 for testing.

# Dataset
n_train = mnist.train.num_examples           # 55,000 training samples
n_validation = mnist.validation.num_examples # 5,000
n_test = mnist.test.num_examples             # 10,000 testing samples


# Neural Network Architecture & Defining placeholders for input data and its targets
# A placeholder is a tensor where data is passed which receives passes along its input after some matrix multiplication.
# Global Variables: Representing the architecture of the neural network including the hidden layers
# input size of the image, its hidden layers with the num of iterations, alongside its output layer, or total batch size.

n_input = 784             # input layer (28x28 pixels)
n_hidden1 = 512           # 1st hidden layer
n_hidden2 = 256           # 2nd hidden layer
n_hidden3 = 128           # 3rd hidden layer
n_output = 10             # output layer (ie 0-9 digits)

# fixed hyperparameters-* vary the range of learning rate bounds in order to observe all three phases. *

learning_rate = 1e-4      # Scales the magnitude of the weight updates in order to minimize the network's loss function.
n_iterations = 1000       # No. of steps through training
batch_size = 128          # No. of training examples at each step
dropout = 0.5             # Individual nodes are ignored and dropped from the net with a
                          # A Bernoulli Distribution with a probability of (1-p) or with a probability of p.
                          # This helps in reducing over-fitting in neural networks by preventing complex
                          # co-adaptations on the training data.
# Similar to L1 and L2; Laplacian and Gaussian penalties, respectively, in Logistic Regression.



# Tensorflow Graph where images are fed into the input X and Y is supplied by the placeholder targets
# X hold the parameter 'None' is a random undefined number of 784-pixel images
# The shape of Y holds 10 possible cases as outputs.
# The Control tensor is a placeholder rather than a immutable variable, it keeps the dropout rate in check.

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
Control = tf.placeholder(tf.float32)            # holding a 32 floating point value


# Weights & Bias parameters of the model are responsible for how well it learns, representing its strength between units
# The weights are initialized as a normally distributed random variable shaping a truncated normal distribution.
# Values are small and near zero, so they are free to sway in each dimension or any direction.
# The input size is shaped into a single tensor vector

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

# The bias is held to a small constant ensuring the tensors are active in the early stages which helps contribute to
# its propagation: errors are computed at the output and distributed backwards throughout the network's layers making
# the biases important, which is why they are stored in a dictionary easy access.

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# Layers of the neural network are used to manipulate tensors in ways of matrix operations.
# Hidden layers preform matrix multiplication on the previous layers output,followed by addition of the bias.
# The final hidden layer is given a drop-out value of 0.5

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, Control)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']


# Loss-function to optimize: cross-entropy or log-loss, essentially the difference between 2 probability distributions.
# (the predictions and the labels). A perfect classification
# would result in a cross-entropy of 0, with the loss completely minimized.

# To reduce the loss function, the gradient descent optimization algorithm is used to find the local min of a function.
# Implemented below is the Adam optimizer which utilizes momentum to speed up the computational process bvy means of
# calculating the exponentially weighted average of the gradients and using that in the adjustments to the network.

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In order to optimize the loss function, Testing and Training the dataset will predict the digits more accurately.
# While the dataset is fed into a graph and keeping track of the correct predictions,
# mini-batches are printed out to check each iteration and check its efficiency.

# The function used compares which images are being predicted correctly by determining the output_layer & Y labels.
# The equal function then returns this as a list of Booleans in order to cast the list into floats allowing the
# computation of the mean and get the models total accuracy.
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# training the networks global variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# The training process occurs in four routines, which are repeated over a set of iterations with the goal
# to minimize the difference between predicted and true labels, with the ultimate aim to optimize the loss function.
# -->  Propagate values forward through the network
# -->  compute the loss
# -->  Propagate values backward through the network
# -->  Update the parameters
# After there is a reduction in loss, and training can be stopped, the network can be used as a model for newer data.

# Train on mini-batches and run the data against the sessions graph
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, Control:dropout})

    # print loss and accuracy (per minibatch) of 100 iterations of training steps. PER BATCH
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, Control:1.0})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

# Once training is complete, you can run the session on the test images.
# Ensuring all units are active in the testing process with a control drop out rate of 1.0
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, Control:1.0})
print("\nAccuracy on test set:", test_accuracy)

# Output shows how accurate our model is.
# In order to become more accurate, many variables can be changed such as the learning rate,
# dropout threshold, batch size, as well as the number of iterations.
# The amount of nodes or hidden layers, are potential ways to see how it affects the models accuracy.

# ---------------------------------------------------------------------------------------------------------------- #
# To test a signle image, a couple libraries are added such as numpy for its arrays and the open function in the Image
# Library which loads the single image in a 4-D array which contains the RGB color scheme and channels.

# The function convert with parameter L allows to reduce the 4D RGBA representation to one grayscale color channel.
# Utilizing numpy, it gets stored in an array which becomes inverted due to the matrix value 0 representing black.

img = np.invert(Image.open("test_img.png").convert('L')).ravel()


# The squeeze function returns the single integer from the array,
# resulting in an output which demonstrates that the network recognizes that the image loaded is indeed a 2.
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))
