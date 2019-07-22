import tensorflow.compat.v1 as tf

# with place holder, constants and multiple graph
# simple_math_operation folder will be created
# to run that open command prompt -> python tensorboard --logdir="foldername"
# tensorboard --logdir="entire\folder\path"
# url will be show -> load that in the browser

# if facing issue in loading the url
# tensorboard --logdir="entire\folder\path" --host localhost --port 8085(anyport)

simplemathwithconstant = tf.Graph()
with simplemathwithconstant.as_default():
    with tf.Session() as smwc:
        a = tf.constant(12, name="var_a")
        b = tf.constant(3, name="var_b")
        c = tf.constant(12, name="var_c")
        d = tf.placeholder(tf.int32, name='var_d')

        mul = tf.multiply(a, b, name="multiply_operation")
        # Error: Deprecated in favor of operator or tf.math.divide.
        # div = tf.div(c, d, name="divide_operation")
        # Error: Tensors in list passed to 'inputs' of 'AddN' Op have types [int32, float64] that don't all match.
        # typecast the output to int
        div = tf.cast(tf.math.divide(c, d, name="divide_operation"), tf.int32)

        addition = tf.add_n([mul, div], name="addition_operation")
        print(smwc.run(addition, feed_dict={d: 4}))
        writer = tf.summary.FileWriter("./simple_math_operation", smwc.graph)
        writer.close()

with tf.Session() as default_graph_session:
    a = tf.constant([1])
    b = tf.constant([3])
    addition = tf.add_n([a, b])
    print(default_graph_session.run(addition))
    writer = tf.summary.FileWriter("./simple_math_operation1", default_graph_session.graph)
    writer.close()


