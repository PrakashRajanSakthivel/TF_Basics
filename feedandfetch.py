import tensorflow.compat.v1 as tf

m = tf.constant([10,12], name="var_a")

x = tf.placeholder(tf.int32, name="ph_a")
c = tf.placeholder(tf.int32, name="ph_b")

mx = tf.multiply(m, x, name="multiple")

fetch1 = tf.add(mx, c, name="Addition")

fetch2 = tf.math.divide(x, c, name="divide")

with tf.Session() as tf_session:
    print(tf_session.run(fetch1, feed_dict={x: [1, 2], c: [2, 3]}))

    # positional param
    print(tf_session.run(fetch2, feed_dict={x: [2, 2], c: [3, 3]}))

    # name param
    print(tf_session.run(fetches=[fetch1, fetch2], feed_dict={x: [1, 2], c: [2, 3]}))
