from __future__ import division
import os

import cv2
import numpy as np
import tensorflow as tf

from TensorflowToolbox.utility import file_loader


class InputLayer(object):
    def __init__(self, file_name, params, is_train):
        self.file_name = file_name
        self.file_parse = file_loader.TextFileLoader()
        self.file_parse.read_file(file_name)
        self.file_len = self.file_parse.get_file_len()
        self.is_train = is_train
        self.params = params

    def _py_read_data(self):
        file_list = self.file_parse.get_next(1)
        image_name, label_name = file_list[0]
        image_name = image_name.encode("utf-8")
        label_name = label_name.encode("utf-8")

        return image_name, label_name #, image, label

    def read_data(self, dtypes):
        return tf.py_func(self._py_read_data, [], dtypes)

    def process_data(self, read_tensor, dtypes):
        image_name = read_tensor[0]
        label_name = read_tensor[1]

        # image = read_tensor[1]
        # label = read_tensor[2]
        # image = tf.image.per_image_standardization(image)
        # if self.is_train:
        #     image = tf.image.random_brightness(image, 0.5)
        # rcrop_size = [self.params.p_img_h, self.params.p_img_w]
        # image, label = self._random_crop(rcrop_size, image, label)

        # return [image_name, image, label]
        return image_name, label_name
