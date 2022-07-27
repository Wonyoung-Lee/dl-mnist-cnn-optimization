from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main


        # TODO: Initialize all hyperparameters

        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # TODO: Initialize all trainable parameters


        self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([5,5,20,20], stddev=0.1))


        self.weight1 = tf.Variable(tf.random.truncated_normal(shape=[320, 150], stddev=0.1, dtype=tf.float32))
        self.weight2 = tf.Variable(tf.random.truncated_normal(shape=[150, 100], stddev=0.1, dtype=tf.float32))
        self.weight3 = tf.Variable(tf.random.truncated_normal(shape=[100, self.num_classes], stddev=0.1, dtype=tf.float32))


        self.bias1 = tf.Variable(tf.random.truncated_normal([150], stddev=0.1, dtype=tf.float32))
        self.bias2 = tf.Variable(tf.random.truncated_normal([100], stddev=0.1, dtype=tf.float32))
        self.bias3 = tf.Variable(tf.random.truncated_normal([self.num_classes], stddev=0.1, dtype=tf.float32))

        self.offset1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1, dtype=tf.float32))
        self.offset2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))
        self.offset3 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))

        self.scale1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1, dtype=tf.float32))
        self.scale2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))
        self.scale3 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))

        self.biasConvolution1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1, dtype=tf.float32))
        self.biasConvolution2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))
        self.biasConvolution3 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))



    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)


        layer1_c = tf.nn.conv2d(inputs, self.filter1, [1,2,2,1], 'SAME')
        layer1_c = tf.nn.bias_add(layer1_c, self.biasConvolution1)

        l1_mean, l1_variance = tf.nn.moments(layer1_c, axes=[0,1,2])

        l1_batch_norm = tf.nn.batch_normalization(layer1_c, l1_mean, l1_variance, self.offset1, self.scale1, variance_epsilon=1e-5)
        relu1 = tf.nn.relu(l1_batch_norm)
        l1_max_pool = tf.nn.max_pool(relu1, 3, 2, 'SAME')


        layer2_c = tf.nn.conv2d(l1_max_pool, self.filter2, [1,1,1,1], 'SAME')
        layer2_c = tf.nn.bias_add(layer2_c, self.biasConvolution2)

        l2_mean, l2_variance = tf.nn.moments(layer2_c, axes=[0,1,2])

        l2_batch_norm = tf.nn.batch_normalization(layer2_c, l2_mean, l2_variance, self.offset2, self.scale2, variance_epsilon=1e-5)
        relu2 = tf.nn.relu(l2_batch_norm)
        l2_max_pool = tf.nn.max_pool(relu2,2,2, 'SAME')


        if is_testing is False:
            layer3_c = tf.nn.conv2d(l2_max_pool, self.filter3, [1,1,1,1], 'SAME')

        else:
            layer3_c = conv2d(l2_max_pool, self.filter3, [1,1,1,1], 'SAME')

        layer3_c = tf.nn.bias_add(layer3_c, self.biasConvolution3)


        l3_mean, l3_variance = tf.nn.moments(layer3_c, axes=[0,1,2])

        l3_batch_norm = tf.nn.batch_normalization(layer3_c, l3_mean, l3_variance, self.offset3, self.scale3, variance_epsilon=1e-5)
        relu3 = tf.nn.relu(l3_batch_norm)

        reshape = tf.reshape(relu3, [relu3.shape[0], -1])

        dl_1 = tf.nn.relu(tf.matmul(reshape, self.weight1) + self.bias1)
        dl_1 = tf.nn.dropout(dl_1, rate=0.3)

        dl_2 = tf.nn.relu(tf.matmul(dl_1, self.weight2) + self.bias2)
        dl_2 = tf.nn.dropout(dl_2, rate=0.3)

        dl_3 = tf.nn.relu(tf.matmul(dl_2, self.weight3) + self.bias3)

        return dl_3

### We have included all 4 loss functions that we will be applying on both MNIST and NIST Datasets. We have left one uncommented.

### Loss Function: softmax_cross_entropy_with_logits
    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        l = tf.nn.softmax_cross_entropy_with_logits(labels, logits)

        return tf.reduce_mean(l)

### Loss Function: sparse_categorical_crossentropy
    #def loss(self, prbs, labels, mask):
          #"""
          #Calculates the total model cross-entropy loss after one forward pass.
          #Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

          #:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
          #:param labels:  integer tensor, word prediction labels [batch_size x window_size]
          #:param mask:  tensor that acts as a padding mask [batch_size x window_size]
          #:return: the loss of the model as a tensor
          #"""

          #loss = tf.keras.metrics.sparse_categorical_crossentropy(labels, prbs)
          #return tf.reduce_sum(loss * mask)

### Loss Function: average_cross_entropy_loss
    #def loss(self, probabilities, labels):
         #"""
          #Calculates the model cross-entropy loss after one forward pass.
         #Loss should be decreasing with every training loop (step).
          #:param probabilities: matrix that contains the probabilities
          #of each class for each image
          #:param labels: the true batch labels
          #:return: average loss per batch element (float)
          #"""

          #prob = -np.log(probabilities[np.arange(self.batch_size), labels])
          #prob = np.sum(prob)/probabilities.shape[0]

          #return prob

### Loss Function: KL-Divergence
    #def bce_function(x_hat, x):
        #"""
        #Computes the reconstruction loss of the VAE.

        #Inputs:
        #- x_hat: Reconstructed input data of shape (N, 1, H, W)
        #- x: Input data for this timestep of shape (N, 1, H, W)

        #Returns:
        #- reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
        #"""
        #bce_fn = tf.keras.losses.BinaryCrossentropy(
        #    from_logits=False,
        #    reduction=tf.keras.losses.Reduction.SUM,
        #)
        #reconstruction_loss = bce_fn(x, x_hat) * x.shape[-1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
        #return reconstruction_loss

    #def loss(x_hat, x, mu, logvar):
        #"""
        #Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
        #Returned loss is the average loss per sample in the current batch.

        #Inputs:
        #- x_hat: Reconstructed input data of shape (N, 1, H, W)
        #- x: Input data for this timestep of shape (N, 1, H, W)
        #- mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        #- logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension

        #Returns:
        #- loss: Tensor containing the scalar loss for the negative variational lowerbound
        #"""
        #loss = None
        ################################################################################################
        # TODO: Compute negative variational lowerbound loss as described in the notebook              #
        ################################################################################################
        # Replace "pass" statement with your code
        #l = -0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
        #loss = (tf.reduce_sum(l) + bce_function(x_hat, x)) / x.shape[0]

        ################################################################################################
        #                            END OF YOUR CODE                                                  #
        ################################################################################################
        return loss





    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    i = tf.random.shuffle(tf.range(0, train_inputs.shape[0]))
    train_inputs = tf.gather(train_inputs, i)
    train_labels = tf.gather(train_labels, i)

    inputs_per_batch = int(train_inputs.shape[0]/model.batch_size)

    for batch in range(inputs_per_batch):

        inputs = tf.image.random_flip_left_right(train_inputs[batch * model.batch_size: (batch + 1) * model.batch_size])
        labels = train_labels[batch * model.batch_size: (batch + 1) * model.batch_size]

        with tf.GradientTape() as tape:
            loss = model.loss(model.call(inputs), labels)

        grad = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.

    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    accuracy = model.accuracy(model.call(test_inputs, is_testing=True), test_labels)

    return accuracy

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    CS1470 students should receive a final accuracy
    on the testing examples for cat and dog of >=70%.

    CS2470 students should receive a final accuracy
    on the testing examples for cat and dog of >=75%.

    :return: None
    '''


    first_class = 3
    second_class = 5

    ### MNIST DATASET (uncomment to test)
    #train_inputs, train_labels = get_data('../../data/train-labels.idx1-ubyte', first_class, second_class)
    #test_inputs, test_labels = get_data('../../data/train-images.idx3-ubyte', first_class, second_class)
    #test_inputs, test_labels = get_data('../../data/t10k-labels.idx1-ubyte', first_class, second_class)
    test_inputs, test_labels = get_data('../../data/t10k-images.idx3-ubyte', first_class, second_class)

    ### NIST DATASET
    test_inputs, test_labels = get_data('../../data/by_class_md5.log', first_class, second_class)


    model = Model()

    for epoch in range(0,10):
        train(model, train_inputs, train_labels)

    print("Test Probability:", test(model, test_inputs, test_labels))
    print('hello')


if __name__ == '__main__':
    main()
