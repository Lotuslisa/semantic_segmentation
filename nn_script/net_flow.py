import os

import cv2
import numpy as np
import tensorflow as tf

from TensorflowToolbox.utility import file_io
from TensorflowToolbox.model_flow import save_func as sf
from TensorflowToolbox.utility import utility_func as uf 
from TensorflowToolbox.utility import image_utility_func as iuf 
from TensorflowToolbox.utility import result_obj as ro

from data_input import DataInput
from data_input_ph import DataInputPh

class NetFlow(object):
    def __init__(self, params, load_train, load_test):
        self.load_train = load_train
        self.load_test = load_test
        self.params = params
        self.train_data_input = None
        self.test_data_input = None
        self.data_ph = DataInputPh(params)
        self.threads = list()

        config_proto = uf.define_graph_config(self.params.gpu_fraction)
        self.sess = tf.Session(config=config_proto)
        self.coord = tf.train.Coordinator()

        if load_train:
            self.train_data_input = DataInput(params, 'train', True, self.sess, self.coord)
        if load_test:
            self.test_data_input = DataInput(params, 'val', False, self.sess, self.coord)
       

        model = file_io.import_module_class(params.model_def_name, "Model")
        #if self.train_data_input is not None: 
        #    self.data_ph = self.tensor_to_palceholder(self.train_data_input)
        #elif self.test_data_input is not None: 
        #    self.data_ph = self.tensor_to_palceholder(self.test_data_input)

        #self.model = model(self.data_ph, params)
        self.model = model(self.data_ph, params)
    
    def tensor_to_placeholder(self, tensors):
        for tensor in tensors:
            ph = tf.placeholder(tensor.dtype, tensor.get_shape().as_list())

        return ph

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


    def demap_to_color(self, demap):
        demap = demap * 10
        demap[demap > 1] = 1
        demap = iuf.norm_image(demap)
        #demap = demap.astype(np.uint8)
        im_color = cv2.applyColorMap(demap, cv2.COLORMAP_JET)
        return im_color


    def save_demap(self, file_names, density_maps, images):
        result_dir = "/home/guanhang/results/"
        for i in range(file_names.shape[0]):
            basename = os.path.basename(file_names[i])
            image = images[i]
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result_name = result_dir + basename.replace('.png','.demap')
            demap_color = self.demap_to_color(density_maps[i])
            concate_img = np.hstack((image, demap_color))
            cv2.imshow('concat', concate_img)
            cv2.waitKey(0)

    def mainloop(self):
        sess = self.sess
        self.init_var(sess)
        params = self.params
        coord = self.coord
        model = self.model

        if self.load_train:
            for i in range(params.max_training_iter):
                feed_dict = self.data_ph.get_feed_dict(
                                            sess,
                                            self.train_data_input.get_batch_tensor())

                _, train_loss_v = sess.run([model.train_op, model.loss], feed_dict=feed_dict)

                if i % params.test_per_iter == 0:
                    feed_dict = self.data_ph.get_feed_dict(
                                            sess,
                                            self.test_data_input.get_batch_tensor())

                    summ_v, test_loss_v = sess.run([self.summ, model.loss], 
                                                    feed_dict=feed_dict)

                    self.sum_writer.add_summary(summ_v, i)
                    print('i: {}, train_loss: {}, test_loss: {}'.format(
                                                i, train_loss_v, test_loss_v))
                if i != 0 and (i % params.save_per_iter == 0 or \
                               i == params.max_training_iter - 1):
                    sf.save_model(sess, self.saver, params.model_dir,i)

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
                feed_dict = self.data_ph.get_feed_dict(sess, batch_tensor_v)

                density_map, test_loss_v = sess.run([model.output, model.loss], 
                                                    feed_dict=feed_dict)

                density_map /= params.density_scale
                batch_file_name = batch_tensor_v[0]
                self.save_demap(batch_file_name, density_map, batch_tensor_v[1])


        coord.request_stop()
        coord.join(self.threads)
