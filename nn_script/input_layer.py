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
        self.file_parse.read_file(file_name, params.shuffle)
        self.file_len = self.file_parse.get_file_len()
        self.is_train = is_train
        self.params = params
        self.data_arg = data_arg.DataArg()

    def _py_read_data(self):
        if self.is_train:
             data_dir = "../segmentation_dataset/"
        else:
             data_dir = ''
   
        #file_list = self.file_parse.get_next(1)
        #image_name, label_name = file_list[0]
        while(1):
            file_list = self.file_parse.get_next(1)
            image_name, label_name = file_list[0]
            if image_name == "/MSRA10K/Imgs/142426.jpg" or label_name == "/MSRA10K/GTs/102052.png":
                continue

            if os.path.exists(data_dir+image_name) and os.path.exists(data_dir+label_name):
                break


        image = cv2.imread(data_dir + image_name)
        label = cv2.imread(data_dir + label_name)
        #back_ground = 1 - label
        #label = np.concatenate((back_ground, label), 2)
        label = np.amax(label, 2)
        
        image = cv2.resize(image, (self.params.r_img_h, self.params.r_img_w)).astype(np.float32)

        image /= 255.0
        label = cv2.resize(label, (self.params.r_label_h, self.params.r_label_w), 
                                   interpolation = cv2.INTER_NEAREST).astype(np.float32)

        if len(image.shape) < 3:
            image = np.expand_dims(image,2)
            image = np.tile(image, (1,1,3))
        
        if len(label.shape) < 3:
            label = np.expand_dims(label, 2)

       # print(label.shape) 
       # label[label > 1] = 1.0  # comment by shz
       # label[label < 1] = 0.0

        # image_name = image_name.encode("utf-8")
        # label_name = label_name.encode("utf-8")

        return image_name, label_name, image, label

    def read_data(self, dtypes):
        return tf.py_func(self._py_read_data, [], dtypes)


#     def ImageCrop(self, im_path, gt_path, im_reshape, transformer):
#         return tf.py_func(self.processImageCrop, [im_path, gt_path, im_reshape,transformer], [float32, float32])
# 
# 
#     
# def processImageCrop(im_path, gt_path, im_reshape, transformer):
# 
#   img_src = caffe.io.load_image(im_path)
#   img_src = perturb(img_src)
#   pathparts = im_path.split('/')
#   gt = caffe.io.load_image(gt_path)
#   gt = gt[:,:,0]
#   gt[gt>0] = 1
#   
#   crop = getRandCrop(img_src.shape[1], img_src.shape[0], 0.9, 1.0)  
#   image_mean = [123.68/255, 116.779/255, 103.939/255]
#   data1 = img_src[crop[1]:crop[3],crop[0]:crop[2],:]
#   data2 = gt[crop[1]:crop[3],crop[0]:crop[2]]
#   if random.random() > 0.5:
#     data1 = data1[:, ::-1,]
#     data2 =data2[:, ::-1,]
#   if random.random() > 0.7: # conver to grey
#     data1 = rgb2gray(data1)
# 
#   data1 = transformer.preprocess('data_in',data1) 
#   data2 = resize_gt(data2, (im_reshape[0]*gt_scale,im_reshape[1]*gt_scale))
# 
#   return data1, data2
       
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
                    rcrop_size = d.pop('rcrop_size')
                    d['ccrop_size'] = rcrop_size
                if "rbright_max" in d:
                    rbright_max = d.pop('rbright_max')
                if "rcontrast_lower" in d:
                    rcontrast_lower = d.pop('rcontrast_lower')
                if "rcontrast_upper" in d:
                    rcontrast_upper = d.pop('rcontrast_upper')
                if "rhue_max" in d:
                    rhue_max = d.pop('rhue_max')
                if "rflip_updown" in d:
                    rflip_updown = d.pop('rflip_updown')
                if "rflipp_leftright" in d:
                    rflipp_leftright = d.pop('rflipp_leftright')

        data_list = read_tensor[2:]
        data_list = self.data_arg(data_list, arg_dict)
        image = data_list[0]
        label = data_list[1]

        return image_name, label_name, image, label
