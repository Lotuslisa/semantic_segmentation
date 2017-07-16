from TensorflowToolbox.data_flow import queue_loader as ql

import input_layer


class DataInput(object):
    def __init__(self, params, partition, is_train, sess, coord):
        if is_train:
            file_name = params.train_file
        else:
            file_name = params.test_file
        i_layer = input_layer.InputLayer(file_name, params, is_train)
        queue_loader = ql.QueueLoader(i_layer, params, is_train, sess, coord)
        self.file_len = i_layer.file_len
        self.threads = queue_loader.thread_list
        self.batch_tensor = queue_loader.batch_data

    def get_image(self):
        return self.batch_tensor[2]

    def get_label(self):
        return self.batch_tensor[3]

    def get_batch_tensor(self):
        return self.batch_tensor

    def get_threads(self):
        return self.threads
