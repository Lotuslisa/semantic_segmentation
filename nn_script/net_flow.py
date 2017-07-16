import os

import cv2
import numpy as np
import tensorflow as tf

from TensorflowToolbox.model_flow import save_func as sf
from TensorflowToolbox.model_flow import model_trainer as mt 
from TensorflowToolbox.utility import file_io
from TensorflowToolbox.utility import utility_func as uf 
from TensorflowToolbox.utility import image_utility_func as iuf 
from TensorflowToolbox.utility import result_obj as ro

from data_input import DataInput


class NetFlow(object):
    def __init__(self, params, load_train, load_test):
        self.load_train = load_train
        self.load_test = load_test
        self.params = params
        self.train_data_input = None
        self.test_data_input = None
        self.threads = list()

        config_proto = uf.define_graph_config(self.params.gpu_fraction)
        self.sess = tf.Session(config=config_proto)
        self.coord = tf.train.Coordinator()

        if load_train:
            self.train_data_input = DataInput(params, 'train', True, self.sess, self.coord)
        if load_test:
            self.test_data_input = DataInput(params, 'val', False, self.sess, self.coord)
       

        model = file_io.import_module_class(params.model_def_name, "Model")
        self.model = model(params)
        self.train_op, self.loss, self.test_loss = mt.model_trainer(
                                         self.model, params.num_gpus,
                                         self.train_data_input,
                                         self.test_data_input)
    
    def init_var(self, sess):
        sf.add_train_var()
        sf.add_loss()
        sf.add_image("image_to_write", 4)
        self.saver = tf.train.Saver()

        if self.load_train:
            self.sum_writer = tf.summary.FileWriter(
                        self.params.train_log_dir, 
                        sess.graph)
            print('write summary')
        self.summ = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        #init_op = tf.initialize_all_variables()

        sess.run(init_op)

        if not self.load_train or self.params.restore_model:
            sf.restore_model(sess, self.saver, self.params.model_dir,
                            self.params.restore_model_name)


    def mainloop(self):
        sess = self.sess
        self.init_var(sess)
        params = self.params
        coord = self.coord
        model = self.model

        if self.load_train:
            for i in range(params.max_training_iter):
                _, train_loss_v = sess.run([self.train_op, self.loss])
                print(train_loss_v)

                # if i % params.test_per_iter == 0:
                #     summ_v, test_loss_v = sess.run([self.summ, self.test_loss])

                #     self.sum_writer.add_summary(summ_v, i)
                #     print('i: {}, train_loss: {}, test_loss: {}'.format(
                #                                 i, train_loss_v, test_loss_v))
                # if i != 0 and (i % params.save_per_iter == 0 or \
                #                i == params.max_training_iter - 1):
                #     sf.save_model(sess, self.saver, params.model_dir,i)

#                    feed_dict = self.get_feed_dict(sess, is_train=False)
#                    loss_v, summ_v, count_diff_v = \
#                                sess.run([self.loss, \
#                                self.summ, self.count_diff], feed_dict)
#                    
#                    tcount_diff_v /= self.desmap_scale
#                    count_diff_v /= self.desmap_scale
#
#                    print_string = "i: %d, train_loss: %.2f, test_loss: %.2f, " \
#                                "train_count_diff: %.2f, test_count_diff: %.2f"%\
#                          (i, tloss_v, loss_v, tcount_diff_v, count_diff_v)
#
#                    print(print_string)
#                    file_io.save_string(print_string, 
#                            self.params["train_log_dir"] + 
#                            self.params["string_log_name"])
#
#                    self.sum_writer.add_summary(summ_v, i)
#                    sf.add_value_sum(self.sum_writer, tloss_v, "train_loss", i)
#                    sf.add_value_sum(self.sum_writer, loss_v, "test_loss", i)
#                    sf.add_value_sum(self.sum_writer, tcount_diff_v, 
#                                                "train_count_diff", i)
#                    sf.add_value_sum(self.sum_writer, count_diff_v, 
#                                                "test_count_diff", i)
#                    #sf.add_value_sum(self.sum_writer, stage2_v, "stage2", i)
#                    #sf.add_value_sum(self.sum_writer, stage3_v, "stage3", i)
#
#                if i != 0 and (i % self.params["save_per_iter"] == 0 or \
#                                i == self.params["max_training_iter"] - 1):
#                    sf.save_model(sess, self.saver, self.params["model_dir"],i)
                    
        else:
            file_len = self.test_data_input.file_len
            batch_size = params.batch_size
            test_iter = int(file_len / batch_size) + 1

            for i in range(test_iter):
                batch_tensor_v = sess.run(self.test_data_input.get_batch_tensor())
                density_map, test_loss_v = sess.run([model.output, model.loss])

                density_map /= params.density_scale
                batch_file_name = batch_tensor_v[0]
                self.save_demap(batch_file_name, density_map, batch_tensor_v[1])


        coord.request_stop()
        coord.join(self.threads)
