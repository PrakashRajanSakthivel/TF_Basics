import tensorflow.compat.v1 as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# One hot notation tells the index of the label like list [i].
# this will read the data from dataset
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# setting the training and test data size from the data we got it
# from the source.
training_digits, training_labels = mnist.train.next_batch(1000)
test_digits, test_labels = mnist.test.next_batch(10)

# setting the place holders to assign the calculate the values.
# since the images is of 28*28, we have 784 pixels and None represents
# dynamic index of the training image.
# mentioning the float to map the grey scale images
training_digits_ph = tf.placeholder("float", [None, 784])
test_digits_ph = tf.placeholder("float", [784])

# calculate the distance L1.
distance = tf.abs(tf.add(training_digits_ph, tf.negative(test_digits_ph)))
c_dist = tf.reduce_sum(distance, axis=1)
prediction = tf.arg_min(c_dist, 0)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # loop over test data
    for i in range(len(test_digits)):
        # Get nearest neighbor
        nn_index = sess.run(prediction,
                            feed_dict={training_digits_ph: training_digits, test_digits_ph: test_digits[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:",
              np.argmax(training_labels[nn_index]),
              "True Label:", np.argmax(test_labels[i]))
