import tensorflow.compat.v1 as tf
import matplotlib.image as mp_image
import matplotlib.pyplot as mp_plot
import os

from PIL import Image

pictures = []
pictures = [os.path.abspath("./Image_Q_Source/pic-1.jpg"),
                os.path.abspath("./Image_Q_Source/pic-2.jpg"),
                os.path.abspath("./Image_Q_Source/pic-3.jpg"),
                os.path.abspath("./Image_Q_Source/pic-4.jpg"),
                os.path.abspath("./Image_Q_Source/pic-5.jpg")]

# Instantiate the queue
queue = tf.train.string_input_producer(pictures)
# instantiate the image file reader
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    # pic1 = mp_image.imread(pictures[0])
    # # pic2 = tf.image.transpose(pic1)
    # # res = sess.run(pic2)
    # mp_plot.imshow(pic1)
    # mp_plot.show()
    image_list = []
    for i in range(len(pictures)):
        _, image_file = image_reader.read(queue)
        image = tf.image.decode_jpeg(image_file)

        image = tf.image.resize_images(image, [224, 224])
        image.set_shape((224, 224, 3))

        image_array = sess.run(image)
        print(image_array.shape)

        Image.fromarray(image_array.astype('uint8'), 'RGB').show()

        image_list.append(tf.expand_dims(image_array, 0))

    coordinator.request_stop()
    coordinator.join(threads)

    index = 0
    writer = tf.summary.FileWriter("./resize_eg", sess.graph)

    for image_tensor in image_list:
        summary_str = sess.run(tf.summary.image("image" + str(index), image_tensor))
        writer.add_summary(summary_str)
        index += 1

    writer.close()





