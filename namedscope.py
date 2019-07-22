import tensorflow.compat.v1 as tf

A = tf.constant(2, tf.int32, name='var_A')
B = tf.constant(3, tf.int32, name='var_B')
C = tf.constant(4, tf.int32, name='var_C')

with tf.name_scope("AplusB"):
    # Addition with the square bracket , default like reduce_sum
    Addition = tf.add_n([A, B])

with tf.name_scope("AMinB"):
    # Subtraction expects the second parameter, y
    Subtraction = tf.math.subtract(C, B)

with tf.name_scope("mul"):
    mul = tf.math.multiply(Addition, Subtraction)

with tf.Session() as m_graph:
    print(m_graph.run(Addition))
    print(m_graph.run(Subtraction))
    print(m_graph.run(mul))
    writer = tf.summary.FileWriter("./namedGraph", m_graph.graph)
    writer.close()
