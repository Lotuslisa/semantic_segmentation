import json

import cv2
import numpy as np
import tensorflow as tf

from TensorflowToolbox.data_flow import queue_loader as ql
import input_layer as il
import params



sess = tf.Session()
coord = tf.train.Coordinator()

file_list = "../file_list/new_file.txt"
i_layer = il.InputLayer(file_list, params, True)

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
    batch_tensor_v = sess.run(batch_tensor)
    
    img = batch_tensor_v[2][0]
    label = batch_tensor_v[3][0]
    # img += np.tile(label[:,:,0, np.newaxis], (1,1,3))
    img += np.tile(label, (1,1,3))
    cv2.imshow("image", img)
    cv2.waitKey(0)

coord.request_stop()
coord.join()
