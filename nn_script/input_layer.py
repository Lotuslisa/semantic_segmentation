from __future__ import division
import os

import cv2
import numpy as np
import tensorflow as tf

from TensorflowToolbox.utility import file_loader
from TensorflowToolbox.data_flow import data_arg


class InputLayer(object):
    def __init__(self, file_name, params, is_train):
        self.file_name = file_name
        self.file_parse = file_loader.TextFileLoader()
        self.file_parse.read_file(file_name)
        self.file_len = self.file_parse.get_file_len()
        self.is_train = is_train
        self.params = params
        self.data_arg = data_arg.DataArg()

    def _py_read_data(self):
        data_dir = "../"
        file_list = self.file_parse.get_next(1)
        image_name, label_name = file_list[0]

        image = cv2.imread(data_dir + image_name)
        label = cv2.imread(data_dir + label_name)
        #back_ground = 1 - label
        #label = np.concatenate((back_ground, label), 2)
        label = np.amax(label, 2)
        
        image = cv2.resize(image, (self.params.r_img_h, self.params.r_img_w)).astype(np.float32)

        image /= 255.0
        label = cv2.resize(label, (self.params.r_label_h, self.params.r_label_w), 
                                   interpolation = cv2.INTER_NEAREST).astype(np.float32)
        if len(label.shape) < 3:
            label = np.expand_dims(label, 2)


        # image_name = image_name.encode("utf-8")
        # label_name = label_name.encode("utf-8")

        return image_name, label_name, image, label

    def read_data(self, dtypes):
        return tf.py_func(self._py_read_data, [], dtypes)

    def process_data(self, read_tensor, dtypes):
        pq_params = self.params.preprocess_queue
        image_name = read_tensor[0]
        label_name = read_tensor[1]
        # image = read_tensor[2]
        # label = read_tensor[3]
        arg_dict = self.params.arg_dict
        if not self.is_train:
            for d in arg_dict:
                if "rcrop_size" in d:
                    rcrop_size = d.pop(rcrop_size)
                    d['ccrp_size'] = rcrop_size

        data_list = read_tensor[2:]
        data_list = self.data_arg(data_list, arg_dict)
        image = data_list[0]
        label = data_list[1]

        return image_name, label_name, image, label
