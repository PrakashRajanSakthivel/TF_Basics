import tensorflow.compat.v1 as tf

hello = tf.constant("Hello TF")
session = tf.Session()
print(session.run(hello))
session.close()

