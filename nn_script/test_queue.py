import json

import cv2
import numpy as np
import tensorflow as tf

from TensorflowToolbox.data_flow import queue_loader as ql
import input_layer as il
import params


def draw_bbox(image, bbox):
    for i in range(bbox.shape[0]):
        if np.sum(bbox[i]) != 0:
            cv2.rectangle(image, (bbox[i][1], bbox[i][0]), (bbox[i][3], bbox[i][2]), (0,0,255))

    return image

sess = tf.Session()
coord = tf.train.Coordinator()

i_layer = il.InputLayer('/home/guanhang/dataset/joint7_veh/val.json', params, True)

#image_name, image, label = i_layer._py_read_data()
##print(image.shape)
##image = draw_bbox(image, label)
##print(image.shape)
#cv2.imshow('img', label)
#cv2.waitKey(0)
#exit(1)

queue_loader = ql.QueueLoader(i_layer, params, True, sess, coord)
batch_tensor = queue_loader.batch_data

init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(10):
    print(sess.run(batch_tensor[0]))

coord.request_stop()
coord.join()
