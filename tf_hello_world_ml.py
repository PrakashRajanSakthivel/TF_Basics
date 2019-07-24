import tensorflow.compat.v1 as tf
import matplotlib.image as mp_image
import matplotlib.pyplot as mp_plot
import os

image = os.path.abspath(".\Image_source\lion_jpeg.jpg")
image_arr = mp_image.imread(image)

print(image_arr.shape)
print(image_arr)
mp_plot.imshow(image_arr)
mp_plot.show()

var_x = tf.Variable(image_arr, name="x")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    transpose = tf.image.transpose(var_x)
    result = sess.run(transpose)
    mp_plot.imshow(result)
    mp_plot.show()



