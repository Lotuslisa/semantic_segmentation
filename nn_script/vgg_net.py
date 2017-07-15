import tensorflow as tf
from TensorflowToolbox.model_flow import model_func as mf
from TensorflowToolbox.utility import image_utility_func as iuf

class Model(object):
    def __init__(self, data_input, params):
        self.data_input = data_input
        self.params = params
        self.model_infer()
        self.model_loss()
        self.model_mini()


    def _deconv2_wrapper(self, b, input_tensor, sample_tensor,
                    output_channel, wd, leaky_param, layer_name):
        #[b, h, w, _] = sample_tensor.get_shape().as_list()
        #[_,_,_,c] = input_tensor.get_shape().as_list()
        [_,_,_,c] = input_tensor.get_shape().as_list()
        [_, h, w, _] = sample_tensor.get_shape().as_list()
        with tf.variable_scope(layer_name):
            #b = tf.shape(input_tensor)[0]
            output_shape = tf.stack([b, h, w, output_channel])

            deconv = mf.deconvolution_2d_layer(input_tensor,
                                [3, 3, output_channel, c],
                                [2, 2], output_shape, 'VALID',
                                wd, 'deconv')
            deconv = mf.add_leaky_relu(deconv, leaky_param)

        return deconv
    
    def model_infer(self):
        image = self.data_input.get_image()

        with tf.variable_scope('input_batch_size'):
            b = tf.shape(image)[0]

        params = self.params
        wd = params.weight_decay
        leaky_param = params.leaky_param
        
        conv11 = mf.add_leaky_relu(mf.convolution_2d_layer(
                        image, [3,3,3,64], [1,1],
                        "SAME", wd, "conv1_1"), leaky_param)
        print(conv11)

        conv12 = mf.add_leaky_relu(mf.convolution_2d_layer(
                        conv11, [3,3,64,64], [1,1],
                        "SAME", wd, "conv1_2"), leaky_param)
        print(conv12)

        conv1_maxpool = mf.maxpool_2d_layer(conv12, [3,3], [2,2], 'maxpool1') 
        
        print(conv1_maxpool)

        conv21 = mf.add_leaky_relu(mf.convolution_2d_layer(
                        conv1_maxpool, [3,3,64,128], [1,1],
                        "SAME", wd, "conv2_1"), leaky_param)

        print(conv21)

        conv22 = mf.add_leaky_relu(mf.convolution_2d_layer(
                        conv21, [3,3,128,128], [1,1],
                        "SAME", wd, "conv2_2"), leaky_param)
        print(conv22)

        conv2_maxpool = mf.maxpool_2d_layer(conv22, [3,3], [2,2], 'maxpool2') 
        print(conv2_maxpool)
        

        conv31 = mf.add_leaky_relu(mf.convolution_2d_layer(
                        conv2_maxpool, [3,3,128,256], [1,1],
                        "SAME", wd, "conv3_1"), leaky_param)
        print(conv31)

        conv32 = mf.add_leaky_relu(mf.convolution_2d_layer(
                        conv31, [3,3,256,256], [1,1],
                        "SAME", wd, "conv3_2"), leaky_param)
        print(conv32)

        conv3_maxpool = mf.maxpool_2d_layer(conv32, [3,3], [2,2], 'maxpool3') 
        print(conv3_maxpool)

        deconv1 =  self._deconv2_wrapper(b, conv3_maxpool, conv31, 256, wd, leaky_param, "deconv1")
        print(deconv1)

        deconv2 =  self._deconv2_wrapper(b, deconv1, conv21, 128, wd, leaky_param, "deconv2")
        print(deconv2)

        deconv3 =  self._deconv2_wrapper(b, deconv2, conv11, 64, wd, leaky_param, "deconv3")
        print(deconv3)

        densmap =  mf.add_leaky_relu(mf.convolution_2d_layer(
                        deconv3, [1, 1, 64, 1], [1, 1],
                        "SAME", wd, "densmap"), leaky_param)

        def _add_image_sum(layer_name):
            with tf.variable_scope(layer_name):
                merged_image = iuf.merge_image(2, [image, densmap])
            tf.add_to_collection("image_to_write", merged_image)

        print(densmap)

        _add_image_sum('result_image')

        self.output = densmap 

    def model_loss(self):
        label = self.data_input.get_label()
        print(label)
        l2_loss = mf.image_l2_loss(self.output, label, "MEAN")
        self.loss = l2_loss
    
    def model_mini(self):
        optimizer = tf.train.AdamOptimizer(
                self.params.init_learning_rate,
                epsilon=1.0)
        self.train_op = optimizer.minimize(self.loss)
