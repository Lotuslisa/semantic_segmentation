from __future__ import division
import os

import cv2
import numpy as np
import tensorflow as tf

from TensorflowToolbox.utility import file_loader


class InputLayer(object):
    def __init__(self, file_name, params, is_train):
        self.file_name = file_name
        self.file_parse = file_loader.JsonFileLoader()
        self.file_parse.read_file(file_name)
        self.file_len = self.file_parse.get_file_len()
        self.is_train = is_train
        self.params = params

    def _resize_bbox(self, bbox, input_size, output_size):
        """
        bbox: [ymin, xmin, ymax, xmax] 
        input_size: [height, width] (1024, 512)
        output_size: (224, 224)
        """
        h_scale = output_size[0] / input_size[0]
        w_scale = output_size[1] / input_size[1]

        for i in range(bbox.shape[0]):
            bbox[i][0] = bbox[i][0] * h_scale
            bbox[i][1] = bbox[i][1] * w_scale
            bbox[i][2] = bbox[i][2] * h_scale
            bbox[i][3] = bbox[i][3] * w_scale
    
    def gauss2d(self, shape=(3,3),sigma=0.5):
        """
            i.e.
            shape = (2,2)
            sigma = 0.5
        """
    
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    
    def add_gauss(self, image, gauss_f):
        h, w = image.shape
        if h > w:
            gauss_f = gauss_f[:w,:w]
            start_i = int((h - w)/2)
            end_i = start_i + w
            image[start_i:end_i,:] += gauss_f
        else:
            gauss_f = gauss_f[:h,:h]
            start_i = int((w - h)/2)
            end_i = start_i + h
            image[:,start_i:end_i] += gauss_f
        return image

    def _box_to_density_map(self, bboxes, image_size):
        density_map = np.zeros(image_size, np.float32)
        for box in bboxes:
            ymin, xmin, ymax, xmax = [int(val) for val in box]
            short_side = max(1, min(ymax - ymin + 1, xmax - xmin + 1))
            sigma = max((short_side/ 2.0)/ 1, 1.0)
            gauss_f = self.gauss2d((short_side, short_side), sigma)
            self.add_gauss(density_map[ymin:ymax+1, xmin:xmax+1], gauss_f)
        return density_map

    def _py_read_data(self):
        file_list = self.file_parse.get_next(1)
        data_dir = os.path.dirname(self.file_name)
        image_dir = file_list[0]['data_dir']

        image_name = os.path.join(data_dir, image_dir, ''.join([file_list[0]['image_id'], '.png']))
        image = cv2.imread(image_name)
        org_image_size = image.shape
        image = cv2.resize(image, (self.params.r_img_w, self.params.r_img_h))
        image = image.astype(np.float32)
        image /= 255.0
        bboxes = np.zeros((self.params.max_num_box, 4), np.float32)
        real_bboxes = np.array(file_list[0]['boxes'], np.float32)
        if real_bboxes.ndim == 1:
            real_bboxes = np.expand_dims(real_bboxes, 0)
        self._resize_bbox(real_bboxes, (org_image_size[0], org_image_size[1]), 
                          (self.params.r_img_h, self.params.r_img_w))

        density_map = self._box_to_density_map(
                                real_bboxes, 
                                (self.params.r_img_h, self.params.r_img_w))
        density_map = np.expand_dims(density_map, 2) 
        label = density_map * self.params.density_scale
        #label[:real_label.shape[0],:] = real_label
        #image_name = image_name.encode('utf-8')
        return image_name, image, label

    def read_data(self):
        dtype = [tf.string, tf.float32, tf.float32]
        return tf.py_func(self._py_read_data, [], dtype)

    def _random_crop(self, rcrop_size, image, label):
        i_height, i_width, i_cha = image.get_shape().as_list()
        offset_height_max = i_height - rcrop_size[0]
        offset_width_max = i_width - rcrop_size[1]
    
        if offset_height_max == 0 and offset_width_max == 0:
            pass
        else:
            r_weight = tf.random_uniform([], 
                        minval = 0, maxval = offset_height_max, 
                        dtype=tf.int32) 
    
            r_width = tf.random_uniform([], 
                        minval = 0, maxval = offset_width_max, 
                        dtype=tf.int32) 
    
            image = tf.image.crop_to_bounding_box(image, 
                        r_weight, r_width, rcrop_size[0], rcrop_size[1])

            label = tf.image.crop_to_bounding_box(label, 
                        r_weight, r_width, rcrop_size[0], rcrop_size[1])

        return image, label

    def process_data(self, read_tensor):
        image_name = read_tensor[0]
        image = read_tensor[1]
        label = read_tensor[2]
        image = tf.image.per_image_standardization(image)
        if self.is_train:
            image = tf.image.random_brightness(image, 0.5)
        rcrop_size = [self.params.p_img_h, self.params.p_img_w]
        image, label = self._random_crop(rcrop_size, image, label)

        return [image_name, image, label]
