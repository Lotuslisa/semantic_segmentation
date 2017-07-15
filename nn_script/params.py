from __future__ import division
import json
import tensorflow as tf

density_scale = 100.0

ratio = 1
r_img_h = int(512 / ratio)
r_img_w = int(1024 / ratio)
r_img_c = 3

ratio = 2
p_img_h = int(512 / ratio)
p_img_w = int(1024 / ratio)
p_img_c = 3

max_num_box = 100
batch_size = 2

read_queue = {
    'capacity':100,
    'dtypes':[tf.string, tf.string],
    'shapes':[[], []],
    #'shapes':[[], [r_img_h, r_img_w, r_img_c], [max_num_box, 4]],
    #'shapes':[[], [r_img_h, r_img_w, r_img_c], [r_img_h, r_img_w, 1]],
    'min_after_dequeue':20,
    'num_threads': 5
}

process_queue = {
    'capacity':100,
    'dtypes':[tf.string, tf.string],
    'shapes':[[], []],
    #'dtypes':[tf.string, tf.float32, tf.float32],
    #'shapes':[[], [p_img_h, p_img_w, p_img_c], [p_img_h, p_img_w, 1]],
    'min_after_dequeue':20,
    'num_threads': 5
}

weight_decay = 0.001
leaky_param = 0.1
init_learning_rate = 0.0001

train_log_dir = '/home/guanhang/summary/'
train_file = '/home/guanhang/dataset/joint7_veh/train.json'
test_file = '/home/guanhang/dataset/joint7_veh/val.json'
gpu_fraction = 0.9
max_training_iter = 100000
save_per_iter = 1000
model_def_name = 'vgg_net'
model_dir = '/home/guanhang/model/'
test_per_iter = 100

restore_model = False
restore_model_name = None

#file_name = 'config.json'
#with open(file_name, 'w') as f:
#    json.dump(params, f, indent=2, separators=(',', ':'))
