from __future__ import division

import json
import tensorflow as tf

r_img_h = 256 
r_img_w = 256
r_img_c = 3

r_label_h = 256
r_label_w = 256
r_label_c = 1

p_img_h = 224
p_img_w = 224
p_img_c = 3

p_label_h = 224
p_label_w = 224
p_label_c = 1

max_num_box = 100
batch_size = 16

load_queue = {
    'capacity':1000,
    'dtypes':[tf.string, tf.string, tf.float32, tf.float32],
    'shapes':[[], [], [r_img_h, r_img_w, r_img_c], [r_label_h, r_label_w, r_label_c]],
    'min_after_dequeue':50,
    'num_threads': 10
}

preprocess_queue = {
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
train_file = '../file_list/blur_train_list.txt'
test_file = '../file_list/blur_test_list.txt'
gpu_fraction = 0.9
max_training_iter = 80000
save_per_iter = 20000
model_def_name = 'vgg_net'
model_dir = '../models/'
test_per_iter = 100
shuffle = True

restore_model = True
restore_model_name = '20170725_0003_iter_79999_model.ckpt'

num_gpus = 1
image_arg_dict = dict()
image_arg_dict["rbright_max"] = 0.2
image_arg_dict["rcontrast_lower"] = 0.5
image_arg_dict["rcontrast_upper"] = 1.5
image_arg_dict["rhue_max"] = 0.2
image_arg_dict["rcrop_size"] = [p_img_h, p_img_w]
image_arg_dict["rflip_updown"] = True
image_arg_dict["rflipp_leftright"] = True

label_arg_dict = dict()
label_arg_dict["rcrop_size"] = [p_label_h, p_label_w]
label_arg_dict["rflip_updown"] = True
label_arg_dict["rflipp_leftright"] = True

arg_dict = list()
arg_dict.append(image_arg_dict)
arg_dict.append(label_arg_dict)

#file_name = 'config.json'
#with open(file_name, 'w') as f:
#    json.dump(params, f, indent=2, separators=(',', ':'))
