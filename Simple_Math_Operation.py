import tensorflow.compat.v1 as tf

a = tf.constant(12, name="var_a")
b = tf.constant(3, name="var_b")
c = tf.constant(12, name="var_c")
d = tf.constant(3, name="var_d")

mul = tf.multiply(a, b, name="multiply_operation")
# Deprecated in favor of operator or tf.math.divide.
# div = tf.div(c, d, name="divide_operation")
# Tensors in list passed to 'inputs' of 'AddN' Op have types [int32, float64] that don't all match.
# typecast the output to int
div = tf.cast(tf.math.divide(c, d, name="divide_operation"), tf.int32)

addition = tf.add_n([mul, div], name="addition_operation")

with tf.Session() as session:
    print(session.run(addition))
    writer = tf.summary.FileWriter("./simple_math_operation", session.graph)
    writer.close()


# simple_math_operation folder will be created
# to run that open command prompt -> python tensorboard --logdir="foldername"
# tensorboard --logdir="C:\Users\adminuser\PycharmProjects\pythontf\simple_math_operation"
# url will be show -> load that in the browser

# if facing issue in loading the url
# tensorboard --logdir="C:\Users\adminuser\PycharmProjects\pythontf\simple_math_operation" --host localhost
# --port 8085(anyport)

