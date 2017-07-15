import tensorflow as tf

from TensorflowToolbox.data_flow import queue_loader as ql

import input_layer


class DataInputPh(object):
    def __init__(self, params):
        with tf.variable_scope('input_placeholder'):
            self.batch_tensor_ph = list()
            for dtype, shape in zip(params.process_queue['dtypes'], 
                                    params.process_queue['shapes']):
                self.batch_tensor_ph.append(tf.placeholder(dtype, [None] + shape))

    def get_image(self):
        return self.batch_tensor_ph[1]

    def get_label(self):
        return self.batch_tensor_ph[2]

    def get_feed_dict(self, sess, feed_tensor):
        feed_dict = dict()
        if isinstance(feed_tensor[0], tf.Tensor):
            feed_tensor_v = sess.run(feed_tensor)
        else:
            feed_tensor_v = feed_tensor

        for tensor_ph, tensor in zip(self.batch_tensor_ph, feed_tensor_v):
            feed_dict[tensor_ph] = tensor
        return feed_dict
