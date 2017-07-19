import tensorflow as tf

from TensorflowToolbox.model_flow import model_func as mf
from TensorflowToolbox.utility import image_utility_func as iuf

import deepcut_new


class Model(object):
    def __init__(self, params):
        self.params = params
        self.scope = 'deepcut'
        self.optimizer = tf.train.AdamOptimizer(
                    params.init_learning_rate,
                    epsilon=1.0)

    def model_infer(self, input_data, is_train):
        params = self.params
        wd = params.weight_decay
        leaky_param = params.leaky_param
        
        image = input_data.get_image()

        logits = deepcut_new.build_graph(image, is_train)

        output = dict()
        output['logits'] = logits
        self.output = output
        return output 

    def model_loss(self, input_data, output):
        label = input_data.get_label()
        logits = output['logits']
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels = label,
                                    logits = logits)
        
        # l2_loss = mf.image_l2_loss(logits, label, "image_l2_loss")
        ### self.loss = l2_loss
        return loss

    def model_optimizer(self):
        return self.optimizer
