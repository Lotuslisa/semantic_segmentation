from __future__ import division

import json
import tensorflow as tf

r_img_h = 224 
r_img_w = 224
r_img_c = 3

r_label_h = 224
r_label_w = 224
r_label_c = 1

p_img_h = 224
p_img_w = 224
p_img_c = 3

p_label_h = 224
p_label_w = 224
p_label_c = 1

max_num_box = 100
batch_size = 2

read_queue = {
    'capacity':1000,
    'dtypes':[tf.string, tf.string, tf.float32, tf.float32],
    'shapes':[[], [], [r_img_h, r_img_w, r_img_c], [r_label_h, r_label_w, r_label_c]],
    'min_after_dequeue':50,
    'num_threads': 10
}

process_queue = {
    'capacity':1000,
    'dtypes':[tf.string, tf.string, tf.float32, tf.float32],
    'shapes':[[], [], [p_img_h, p_img_w, p_img_c], [p_label_h, p_label_w, p_label_c]],
    'min_after_dequeue':50,
    'num_threads': 10
}

weight_decay = 0.001
leaky_param = 0.1
init_learning_rate = 0.0001

train_log_dir = '../logs/'
train_file = '../file_list/new_file.txt'
gpu_fraction = 0.9
max_training_iter = 10
save_per_iter = 1000
model_def_name = 'vgg_net'
model_dir = '../models/'
test_per_iter = 100

restore_model = False
restore_model_name = None

num_gpus = 0

#file_name = 'config.json'
#with open(file_name, 'w') as f:
#    json.dump(params, f, indent=2, separators=(',', ':'))
