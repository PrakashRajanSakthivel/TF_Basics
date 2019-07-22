import tensorflow.compat.v1 as tf

hello = tf.constant("Hello TF")

with tf.Session() as session:
    print(session.run(hello))


