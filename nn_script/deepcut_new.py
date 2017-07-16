import tensorflow as tf
import numpy as np

def add_relu(input_layer):
    return tf.nn.relu(input_layer) 

def conv(input_layer, channel_num, kernel_size, stride, padding, biased=False, relu=False, name='conv', dilation=1):
    if padding is None:
        padding = 'SAME'
    conv_layer = tf.layers.conv2d(
                            inputs=input_layer, 
                            filters=channel_num,
                            kernel_size=kernel_size,
                            strides=(stride, stride),
                            padding=padding,
                            use_bias=biased,
                            name=name,
                            dilation_rate=(dilation, dilation))

    if relu:
        conv_layer = add_relu(conv_layer)

    return conv_layer

def batch_norm(input_layer, is_training, relu, name):
    #return input_layer
    is_training = False
    batch_norm_layer = tf.layers.batch_normalization(
                                inputs=input_layer,
                                training=is_training,
                                name=name)
    #batch_norm_layer = input_layer
    if relu:
        batch_norm_layer=add_relu(batch_norm_layer)
    return batch_norm_layer 

def upsample(input_layer, factor, name):
    size_4D = input_layer.get_shape().as_list()
    out_height = size_4D[1] * factor
    out_width = size_4D[2] * factor
    outputs = tf.image.resize_bilinear(
                        images=input_layer,
                        size=[out_height, out_width],
                        align_corners=None,
                        name=name)
    return outputs

def max_pool(input_layer, ksize, strides, padding, name):
    outputs = tf.nn.max_pool(
              value=input_layer,
              ksize= [1, ksize, ksize, 1],
              strides=[1, strides, strides, 1],
              padding=padding,
              name=name)
    return outputs


def avg_pool(input_layer, ksize, strides, padding, name):
    outputs = tf.nn.avg_pool(
              value = input_layer,
              ksize= [1, ksize, ksize, 1],
              strides=[1, strides, strides, 1],
              padding = padding,
              name = name)
    return outputs


def concat(*arg, **kwargs):
    concat_dim = arg[-1]
    name = kwargs['name']
    concat_layer = tf.concat(arg[:-1], concat_dim, name)
    return concat_layer    


def softmax(input_layer, name):
    return tf.nn.softmax(input_layer, name=name)


def build_graph(batch_data, is_training=False):
    """
    The following neural network structure is automatically converted
    from caffemodel by [caffemodel_converter](https://github.com/jhyume/caffemodel_converter).
    """
    tensors = dict()

    tensors['data'] = batch_data
    # tensors['data'] = tf.constant(np.transpose(np.load('caffe_input.npy'), (0,2,3,1)))

    
    tensors['conv1_7x7_s2_p'] = conv(tensors['data'], 64, 7, 1, padding='SAME', biased=False, relu=False, name='conv1_7x7_s2_p')

    tensors['conv1_7x7_s2_bn_p'] = batch_norm(tensors['conv1_7x7_s2_p'], is_training=is_training, relu=True, name='conv1_7x7_s2_bn_p')

    tensors['conv1_7x7_s2'] = conv(tensors['data'], 64, 7, 2, padding='SAME', biased=False, relu=False, name='conv1_7x7_s2')
    tensors['conv1_7x7_s2_bn'] = batch_norm(tensors['conv1_7x7_s2'], is_training=is_training, relu=True, name='conv1_7x7_s2_bn')

    tensors['pool1_3x3_s2'] = max_pool(tensors['conv1_7x7_s2_bn'], 3, 2, padding='SAME', name='pool1_3x3_s2')
    tensors['conv2_3x3_reduce'] = conv(tensors['pool1_3x3_s2'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_3x3_reduce')
    tensors['conv2_3x3_reduce_bn'] = batch_norm(tensors['conv2_3x3_reduce'], is_training=is_training, relu=True, name='conv2_3x3_reduce_bn')
    tensors['conv2_3x3'] = conv(tensors['conv2_3x3_reduce_bn'], 192, 3, 1, padding='SAME', biased=False, relu=False, name='conv2_3x3')
    tensors['conv2_3x3_bn'] = batch_norm(tensors['conv2_3x3'], is_training=is_training, relu=True, name='conv2_3x3_bn')
    tensors['pool2_3x3_s2'] = max_pool(tensors['conv2_3x3_bn'], 3, 2, padding='SAME', name='pool2_3x3_s2')
    tensors['inception_3a_1x1'] = conv(tensors['pool2_3x3_s2'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3a_1x1')
    tensors['inception_3a_1x1_bn'] = batch_norm(tensors['inception_3a_1x1'], is_training=is_training, relu=True, name='inception_3a_1x1_bn')

    tensors['inception_3a_3x3_reduce'] = conv(tensors['pool2_3x3_s2'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3a_3x3_reduce')
    tensors['inception_3a_3x3_reduce_bn'] = batch_norm(tensors['inception_3a_3x3_reduce'], is_training=is_training, relu=True, name='inception_3a_3x3_reduce_bn')
    tensors['inception_3a_3x3'] = conv(tensors['inception_3a_3x3_reduce_bn'], 64, 3, 1, padding='SAME', biased=False, relu=False, name='inception_3a_3x3')
    tensors['inception_3a_3x3_bn'] = batch_norm(tensors['inception_3a_3x3'], is_training=is_training, relu=True, name='inception_3a_3x3_bn')

    tensors['inception_3a_double3x3_reduce'] = conv(tensors['pool2_3x3_s2'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3a_double3x3_reduce')
    tensors['inception_3a_double3x3_reduce_bn'] = batch_norm(tensors['inception_3a_double3x3_reduce'], is_training=is_training, relu=True, name='inception_3a_double3x3_reduce_bn')
    tensors['inception_3a_double3x3a'] = conv(tensors['inception_3a_double3x3_reduce_bn'], 96, 3, 1, padding='SAME', biased=False, relu=False, name='inception_3a_double3x3a')
    tensors['inception_3a_double3x3a_bn'] = batch_norm(tensors['inception_3a_double3x3a'], is_training=is_training, relu=True, name='inception_3a_double3x3a_bn')
    tensors['inception_3a_double3x3b'] = conv(tensors['inception_3a_double3x3a_bn'], 96, 3, 1, padding='SAME', biased=False, relu=False, name='inception_3a_double3x3b')
    tensors['inception_3a_double3x3b_bn'] = batch_norm(tensors['inception_3a_double3x3b'], is_training=is_training, relu=True, name='inception_3a_double3x3b_bn')

    tensors['inception_3a_pool'] = avg_pool(tensors['pool2_3x3_s2'], 3, 1, padding='SAME', name='inception_3a_pool')
    tensors['inception_3a_pool_proj'] = conv(tensors['inception_3a_pool'], 32, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3a_pool_proj')
    tensors['inception_3a_pool_proj_bn'] = batch_norm(tensors['inception_3a_pool_proj'], is_training=is_training, relu=True, name='inception_3a_pool_proj_bn')

    tensors['inception_3a_output'] = concat(tensors['inception_3a_1x1_bn'], tensors['inception_3a_3x3_bn'], tensors['inception_3a_double3x3b_bn'], tensors['inception_3a_pool_proj_bn'], 3, name='inception_3a_output')
    tensors['inception_3b_1x1'] = conv(tensors['inception_3a_output'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3b_1x1')
    tensors['inception_3b_1x1_bn'] = batch_norm(tensors['inception_3b_1x1'], is_training=is_training, relu=True, name='inception_3b_1x1_bn')

    tensors['inception_3b_3x3_reduce'] = conv(tensors['inception_3a_output'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3b_3x3_reduce')
    tensors['inception_3b_3x3_reduce_bn'] = batch_norm(tensors['inception_3b_3x3_reduce'], is_training=is_training, relu=True, name='inception_3b_3x3_reduce_bn')
    tensors['inception_3b_3x3'] = conv(tensors['inception_3b_3x3_reduce_bn'], 96, 3, 1, padding='SAME', biased=False, relu=False, name='inception_3b_3x3')
    tensors['inception_3b_3x3_bn'] = batch_norm(tensors['inception_3b_3x3'], is_training=is_training, relu=True, name='inception_3b_3x3_bn')

    tensors['inception_3b_double3x3_reduce'] = conv(tensors['inception_3a_output'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3b_double3x3_reduce')
    tensors['inception_3b_double3x3_reduce_bn'] = batch_norm(tensors['inception_3b_double3x3_reduce'], is_training=is_training, relu=True, name='inception_3b_double3x3_reduce_bn')
    tensors['inception_3b_double3x3a'] = conv(tensors['inception_3b_double3x3_reduce_bn'], 96, 3, 1, padding='SAME', biased=False, relu=False, name='inception_3b_double3x3a')
    tensors['inception_3b_double3x3a_bn'] = batch_norm(tensors['inception_3b_double3x3a'], is_training=is_training, relu=True, name='inception_3b_double3x3a_bn')
    tensors['inception_3b_double3x3b'] = conv(tensors['inception_3b_double3x3a_bn'], 96, 3, 1, padding='SAME', biased=False, relu=False, name='inception_3b_double3x3b')
    tensors['inception_3b_double3x3b_bn'] = batch_norm(tensors['inception_3b_double3x3b'], is_training=is_training, relu=True, name='inception_3b_double3x3b_bn')

    tensors['inception_3b_pool'] = avg_pool(tensors['inception_3a_output'], 3, 1, padding='SAME', name='inception_3b_pool')
    tensors['inception_3b_pool_proj'] = conv(tensors['inception_3b_pool'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3b_pool_proj')
    tensors['inception_3b_pool_proj_bn'] = batch_norm(tensors['inception_3b_pool_proj'], is_training=is_training, relu=True, name='inception_3b_pool_proj_bn')

    tensors['inception_3b_output'] = concat(tensors['inception_3b_1x1_bn'], tensors['inception_3b_3x3_bn'], tensors['inception_3b_double3x3b_bn'], tensors['inception_3b_pool_proj_bn'], 3, name='inception_3b_output')
    tensors['inception_3c_3x3_reduce'] = conv(tensors['inception_3b_output'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3c_3x3_reduce')
    tensors['inception_3c_3x3_reduce_bn'] = batch_norm(tensors['inception_3c_3x3_reduce'], is_training=is_training, relu=True, name='inception_3c_3x3_reduce_bn')
    tensors['inception_3c_3x3'] = conv(tensors['inception_3c_3x3_reduce_bn'], 160, 3, 2, padding='SAME', biased=False, relu=False, name='inception_3c_3x3')
    tensors['inception_3c_3x3_bn'] = batch_norm(tensors['inception_3c_3x3'], is_training=is_training, relu=True, name='inception_3c_3x3_bn')

    tensors['inception_3c_double3x3_reduce'] = conv(tensors['inception_3b_output'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_3c_double3x3_reduce')
    tensors['inception_3c_double3x3_reduce_bn'] = batch_norm(tensors['inception_3c_double3x3_reduce'], is_training=is_training, relu=True, name='inception_3c_double3x3_reduce_bn')
    tensors['inception_3c_double3x3a'] = conv(tensors['inception_3c_double3x3_reduce_bn'], 96, 3, 1, padding='SAME', biased=False, relu=False, name='inception_3c_double3x3a')
    tensors['inception_3c_double3x3a_bn'] = batch_norm(tensors['inception_3c_double3x3a'], is_training=is_training, relu=True, name='inception_3c_double3x3a_bn')
    tensors['inception_3c_double3x3b'] = conv(tensors['inception_3c_double3x3a_bn'], 96, 3, 2, padding='SAME', biased=False, relu=False, name='inception_3c_double3x3b')
    tensors['inception_3c_double3x3b_bn'] = batch_norm(tensors['inception_3c_double3x3b'], is_training=is_training, relu=True, name='inception_3c_double3x3b_bn')

    tensors['inception_3c_pool'] = max_pool(tensors['inception_3b_output'], 3, 2, padding='SAME', name='inception_3c_pool')

    tensors['inception_3c_output'] = concat(tensors['inception_3c_pool'], tensors['inception_3c_3x3_bn'], tensors['inception_3c_double3x3b_bn'], 3, name='inception_3c_output')
    tensors['inception_4a_1x1'] = conv(tensors['inception_3c_output'], 224, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4a_1x1')
    tensors['inception_4a_1x1_bn'] = batch_norm(tensors['inception_4a_1x1'], is_training=is_training, relu=True, name='inception_4a_1x1_bn')

    tensors['inception_4a_3x3_reduce'] = conv(tensors['inception_3c_output'], 64, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4a_3x3_reduce')
    tensors['inception_4a_3x3_reduce_bn'] = batch_norm(tensors['inception_4a_3x3_reduce'], is_training=is_training, relu=True, name='inception_4a_3x3_reduce_bn')
    tensors['inception_4a_3x3'] = conv(tensors['inception_4a_3x3_reduce_bn'], 96, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4a_3x3')
    tensors['inception_4a_3x3_bn'] = batch_norm(tensors['inception_4a_3x3'], is_training=is_training, relu=True, name='inception_4a_3x3_bn')

    tensors['inception_4a_double3x3_reduce'] = conv(tensors['inception_3c_output'], 96, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4a_double3x3_reduce')
    tensors['inception_4a_double3x3_reduce_bn'] = batch_norm(tensors['inception_4a_double3x3_reduce'], is_training=is_training, relu=True, name='inception_4a_double3x3_reduce_bn')
    tensors['inception_4a_double3x3a'] = conv(tensors['inception_4a_double3x3_reduce_bn'], 128, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4a_double3x3a')
    tensors['inception_4a_double3x3a_bn'] = batch_norm(tensors['inception_4a_double3x3a'], is_training=is_training, relu=True, name='inception_4a_double3x3a_bn')
    tensors['inception_4a_double3x3b'] = conv(tensors['inception_4a_double3x3a_bn'], 128, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4a_double3x3b')
    tensors['inception_4a_double3x3b_bn'] = batch_norm(tensors['inception_4a_double3x3b'], is_training=is_training, relu=True, name='inception_4a_double3x3b_bn')

    tensors['inception_4a_pool'] = avg_pool(tensors['inception_3c_output'], 3, 1, padding='SAME', name='inception_4a_pool')
    tensors['inception_4a_pool_proj'] = conv(tensors['inception_4a_pool'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4a_pool_proj')
    tensors['inception_4a_pool_proj_bn'] = batch_norm(tensors['inception_4a_pool_proj'], is_training=is_training, relu=True, name='inception_4a_pool_proj_bn')

    tensors['inception_4a_output'] = concat(tensors['inception_4a_1x1_bn'], tensors['inception_4a_3x3_bn'], tensors['inception_4a_double3x3b_bn'], tensors['inception_4a_pool_proj_bn'], 3, name='inception_4a_output')
    tensors['inception_4b_1x1'] = conv(tensors['inception_4a_output'], 192, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4b_1x1')
    tensors['inception_4b_1x1_bn'] = batch_norm(tensors['inception_4b_1x1'], is_training=is_training, relu=True, name='inception_4b_1x1_bn')

    tensors['inception_4b_3x3_reduce'] = conv(tensors['inception_4a_output'], 96, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4b_3x3_reduce')
    tensors['inception_4b_3x3_reduce_bn'] = batch_norm(tensors['inception_4b_3x3_reduce'], is_training=is_training, relu=True, name='inception_4b_3x3_reduce_bn')
    tensors['inception_4b_3x3'] = conv(tensors['inception_4b_3x3_reduce_bn'], 128, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4b_3x3')
    tensors['inception_4b_3x3_bn'] = batch_norm(tensors['inception_4b_3x3'], is_training=is_training, relu=True, name='inception_4b_3x3_bn')

    tensors['inception_4b_double3x3_reduce'] = conv(tensors['inception_4a_output'], 96, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4b_double3x3_reduce')
    tensors['inception_4b_double3x3_reduce_bn'] = batch_norm(tensors['inception_4b_double3x3_reduce'], is_training=is_training, relu=True, name='inception_4b_double3x3_reduce_bn')
    tensors['inception_4b_double3x3a'] = conv(tensors['inception_4b_double3x3_reduce_bn'], 128, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4b_double3x3a')
    tensors['inception_4b_double3x3a_bn'] = batch_norm(tensors['inception_4b_double3x3a'], is_training=is_training, relu=True, name='inception_4b_double3x3a_bn')
    tensors['inception_4b_double3x3b'] = conv(tensors['inception_4b_double3x3a_bn'], 128, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4b_double3x3b')
    tensors['inception_4b_double3x3b_bn'] = batch_norm(tensors['inception_4b_double3x3b'], is_training=is_training, relu=True, name='inception_4b_double3x3b_bn')

    tensors['inception_4b_pool'] = avg_pool(tensors['inception_4a_output'], 3, 1, padding='SAME', name='inception_4b_pool')
    tensors['inception_4b_pool_proj'] = conv(tensors['inception_4b_pool'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4b_pool_proj')
    tensors['inception_4b_pool_proj_bn'] = batch_norm(tensors['inception_4b_pool_proj'], is_training=is_training, relu=True, name='inception_4b_pool_proj_bn')

    tensors['inception_4b_output'] = concat(tensors['inception_4b_1x1_bn'], tensors['inception_4b_3x3_bn'], tensors['inception_4b_double3x3b_bn'], tensors['inception_4b_pool_proj_bn'], 3, name='inception_4b_output')
    tensors['inception_4c_1x1'] = conv(tensors['inception_4b_output'], 160, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4c_1x1')
    tensors['inception_4c_1x1_bn'] = batch_norm(tensors['inception_4c_1x1'], is_training=is_training, relu=True, name='inception_4c_1x1_bn')

    tensors['inception_4c_3x3_reduce'] = conv(tensors['inception_4b_output'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4c_3x3_reduce')
    tensors['inception_4c_3x3_reduce_bn'] = batch_norm(tensors['inception_4c_3x3_reduce'], is_training=is_training, relu=True, name='inception_4c_3x3_reduce_bn')
    tensors['inception_4c_3x3'] = conv(tensors['inception_4c_3x3_reduce_bn'], 160, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4c_3x3')
    tensors['inception_4c_3x3_bn'] = batch_norm(tensors['inception_4c_3x3'], is_training=is_training, relu=True, name='inception_4c_3x3_bn')

    tensors['inception_4c_double3x3_reduce'] = conv(tensors['inception_4b_output'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4c_double3x3_reduce')
    tensors['inception_4c_double3x3_reduce_bn'] = batch_norm(tensors['inception_4c_double3x3_reduce'], is_training=is_training, relu=True, name='inception_4c_double3x3_reduce_bn')
    tensors['inception_4c_double3x3a'] = conv(tensors['inception_4c_double3x3_reduce_bn'], 160, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4c_double3x3a')
    tensors['inception_4c_double3x3a_bn'] = batch_norm(tensors['inception_4c_double3x3a'], is_training=is_training, relu=True, name='inception_4c_double3x3a_bn')
    tensors['inception_4c_double3x3b'] = conv(tensors['inception_4c_double3x3a_bn'], 160, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4c_double3x3b')
    tensors['inception_4c_double3x3b_bn'] = batch_norm(tensors['inception_4c_double3x3b'], is_training=is_training, relu=True, name='inception_4c_double3x3b_bn')

    tensors['inception_4c_pool'] = avg_pool(tensors['inception_4b_output'], 3, 1, padding='SAME', name='inception_4c_pool')
    tensors['inception_4c_pool_proj'] = conv(tensors['inception_4c_pool'], 96, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4c_pool_proj')
    tensors['inception_4c_pool_proj_bn'] = batch_norm(tensors['inception_4c_pool_proj'], is_training=is_training, relu=True, name='inception_4c_pool_proj_bn')

    tensors['inception_4c_output'] = concat(tensors['inception_4c_1x1_bn'], tensors['inception_4c_3x3_bn'], tensors['inception_4c_double3x3b_bn'], tensors['inception_4c_pool_proj_bn'], 3, name='inception_4c_output')
    tensors['inception_4d_1x1'] = conv(tensors['inception_4c_output'], 96, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4d_1x1')
    tensors['inception_4d_1x1_bn'] = batch_norm(tensors['inception_4d_1x1'], is_training=is_training, relu=True, name='inception_4d_1x1_bn')

    tensors['inception_4d_3x3_reduce'] = conv(tensors['inception_4c_output'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4d_3x3_reduce')
    tensors['inception_4d_3x3_reduce_bn'] = batch_norm(tensors['inception_4d_3x3_reduce'], is_training=is_training, relu=True, name='inception_4d_3x3_reduce_bn')
    tensors['inception_4d_3x3'] = conv(tensors['inception_4d_3x3_reduce_bn'], 192, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4d_3x3')
    tensors['inception_4d_3x3_bn'] = batch_norm(tensors['inception_4d_3x3'], is_training=is_training, relu=True, name='inception_4d_3x3_bn')

    tensors['inception_4d_double3x3_reduce'] = conv(tensors['inception_4c_output'], 160, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4d_double3x3_reduce')
    tensors['inception_4d_double3x3_reduce_bn'] = batch_norm(tensors['inception_4d_double3x3_reduce'], is_training=is_training, relu=True, name='inception_4d_double3x3_reduce_bn')
    tensors['inception_4d_double3x3a'] = conv(tensors['inception_4d_double3x3_reduce_bn'], 192, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4d_double3x3a')
    tensors['inception_4d_double3x3a_bn'] = batch_norm(tensors['inception_4d_double3x3a'], is_training=is_training, relu=True, name='inception_4d_double3x3a_bn')
    tensors['inception_4d_double3x3b'] = conv(tensors['inception_4d_double3x3a_bn'], 192, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4d_double3x3b')
    tensors['inception_4d_double3x3b_bn'] = batch_norm(tensors['inception_4d_double3x3b'], is_training=is_training, relu=True, name='inception_4d_double3x3b_bn')

    tensors['inception_4d_pool'] = avg_pool(tensors['inception_4c_output'], 3, 1, padding='SAME', name='inception_4d_pool')
    tensors['inception_4d_pool_proj'] = conv(tensors['inception_4d_pool'], 96, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4d_pool_proj')
    tensors['inception_4d_pool_proj_bn'] = batch_norm(tensors['inception_4d_pool_proj'], is_training=is_training, relu=True, name='inception_4d_pool_proj_bn')

    tensors['inception_4d_output'] = concat(tensors['inception_4d_1x1_bn'], tensors['inception_4d_3x3_bn'], tensors['inception_4d_double3x3b_bn'], tensors['inception_4d_pool_proj_bn'], 3, name='inception_4d_output')
    tensors['inception_4e_3x3_reduce'] = conv(tensors['inception_4d_output'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4e_3x3_reduce')
    tensors['inception_4e_3x3_reduce_bn'] = batch_norm(tensors['inception_4e_3x3_reduce'], is_training=is_training, relu=True, name='inception_4e_3x3_reduce_bn')
    tensors['inception_4e_3x3'] = conv(tensors['inception_4e_3x3_reduce_bn'], 192, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4e_3x3')
    tensors['inception_4e_3x3_bn'] = batch_norm(tensors['inception_4e_3x3'], is_training=is_training, relu=True, name='inception_4e_3x3_bn')

    tensors['inception_4e_double3x3_reduce'] = conv(tensors['inception_4d_output'], 192, 1, 1, padding='SAME', biased=False, relu=False, name='inception_4e_double3x3_reduce')
    tensors['inception_4e_double3x3_reduce_bn'] = batch_norm(tensors['inception_4e_double3x3_reduce'], is_training=is_training, relu=True, name='inception_4e_double3x3_reduce_bn')
    tensors['inception_4e_double3x3a'] = conv(tensors['inception_4e_double3x3_reduce_bn'], 256, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4e_double3x3a')
    tensors['inception_4e_double3x3a_bn'] = batch_norm(tensors['inception_4e_double3x3a'], is_training=is_training, relu=True, name='inception_4e_double3x3a_bn')
    tensors['inception_4e_double3x3b'] = conv(tensors['inception_4e_double3x3a_bn'], 256, 3, 1, padding='SAME', biased=False, relu=False, name='inception_4e_double3x3b')
    tensors['inception_4e_double3x3b_bn'] = batch_norm(tensors['inception_4e_double3x3b'], is_training=is_training, relu=True, name='inception_4e_double3x3b_bn')

    tensors['inception_4e_pool'] = max_pool(tensors['inception_4d_output'], 3, 1, padding='SAME', name='inception_4e_pool')

    tensors['inception_4e_output'] = concat(tensors['inception_4e_pool'], tensors['inception_4e_3x3_bn'], tensors['inception_4e_double3x3b_bn'], 3, name='inception_4e_output')
    tensors['inception_5a_1x1'] = conv(tensors['inception_4e_output'], 352, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5a_1x1')
    tensors['inception_5a_1x1_bn'] = batch_norm(tensors['inception_5a_1x1'], is_training=is_training, relu=True, name='inception_5a_1x1_bn')

    tensors['inception_5a_3x3_reduce'] = conv(tensors['inception_4e_output'], 192, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5a_3x3_reduce')
    tensors['inception_5a_3x3_reduce_bn'] = batch_norm(tensors['inception_5a_3x3_reduce'], is_training=is_training, relu=True, name='inception_5a_3x3_reduce_bn')
    tensors['inception_5a_3x3'] = conv(tensors['inception_5a_3x3_reduce_bn'], 320, 3, 1, padding=None, biased=False, relu=False, name='inception_5a_3x3', dilation = 2)
    tensors['inception_5a_3x3_bn'] = batch_norm(tensors['inception_5a_3x3'], is_training=is_training, relu=True, name='inception_5a_3x3_bn')

    tensors['inception_5a_double3x3_reduce'] = conv(tensors['inception_4e_output'], 160, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5a_double3x3_reduce')
    tensors['inception_5a_double3x3_reduce_bn'] = batch_norm(tensors['inception_5a_double3x3_reduce'], is_training=is_training, relu=True, name='inception_5a_double3x3_reduce_bn')
    tensors['inception_5a_double3x3a'] = conv(tensors['inception_5a_double3x3_reduce_bn'], 224, 3, 1, padding=None, biased=False, relu=False, name='inception_5a_double3x3a', dilation = 2)
    tensors['inception_5a_double3x3a_bn'] = batch_norm(tensors['inception_5a_double3x3a'], is_training=is_training, relu=True, name='inception_5a_double3x3a_bn')
    tensors['inception_5a_double3x3b'] = conv(tensors['inception_5a_double3x3a_bn'], 224, 3, 1, padding=None, biased=False, relu=False, name='inception_5a_double3x3b', dilation =2)
    tensors['inception_5a_double3x3b_bn'] = batch_norm(tensors['inception_5a_double3x3b'], is_training=is_training, relu=True, name='inception_5a_double3x3b_bn')

    tensors['inception_5a_pool'] = avg_pool(tensors['inception_4e_output'], 5, 1, padding='SAME', name='inception_5a_pool')
    tensors['inception_5a_pool_proj'] = conv(tensors['inception_5a_pool'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5a_pool_proj')
    tensors['inception_5a_pool_proj_bn'] = batch_norm(tensors['inception_5a_pool_proj'], is_training=is_training, relu=True, name='inception_5a_pool_proj_bn')

    tensors['inception_5a_output'] = concat(tensors['inception_5a_1x1_bn'], tensors['inception_5a_3x3_bn'], tensors['inception_5a_double3x3b_bn'], tensors['inception_5a_pool_proj_bn'], 3, name='inception_5a_output')
    tensors['inception_5b_1x1'] = conv(tensors['inception_5a_output'], 352, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5b_1x1')
    tensors['inception_5b_1x1_bn'] = batch_norm(tensors['inception_5b_1x1'], is_training=is_training, relu=True, name='inception_5b_1x1_bn')

    tensors['inception_5b_3x3_reduce'] = conv(tensors['inception_5a_output'], 192, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5b_3x3_reduce')
    tensors['inception_5b_3x3_reduce_bn'] = batch_norm(tensors['inception_5b_3x3_reduce'], is_training=is_training, relu=True, name='inception_5b_3x3_reduce_bn')
    tensors['inception_5b_3x3'] = conv(tensors['inception_5b_3x3_reduce_bn'], 320, 3, 1, padding=None, biased=False, relu=False, name='inception_5b_3x3', dilation = 2)
    tensors['inception_5b_3x3_bn'] = batch_norm(tensors['inception_5b_3x3'], is_training=is_training, relu=True, name='inception_5b_3x3_bn')

    tensors['inception_5b_double3x3_reduce'] = conv(tensors['inception_5a_output'], 192, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5b_double3x3_reduce')
    tensors['inception_5b_double3x3_reduce_bn'] = batch_norm(tensors['inception_5b_double3x3_reduce'], is_training=is_training, relu=True, name='inception_5b_double3x3_reduce_bn')
    tensors['inception_5b_double3x3a'] = conv(tensors['inception_5b_double3x3_reduce_bn'], 224, 3, 1, padding=None, biased=False, relu=False, name='inception_5b_double3x3a', dilation = 2)
    tensors['inception_5b_double3x3a_bn'] = batch_norm(tensors['inception_5b_double3x3a'], is_training=is_training, relu=True, name='inception_5b_double3x3a_bn')
    tensors['inception_5b_double3x3b'] = conv(tensors['inception_5b_double3x3a_bn'], 224, 3, 1, padding=None, biased=False, relu=False, name='inception_5b_double3x3b', dilation = 2)
    tensors['inception_5b_double3x3b_bn'] = batch_norm(tensors['inception_5b_double3x3b'], is_training=is_training, relu=True, name='inception_5b_double3x3b_bn')

    tensors['inception_5b_pool'] = max_pool(tensors['inception_5a_output'], 5, 1, padding='SAME', name='inception_5b_pool')
    tensors['inception_5b_pool_proj'] = conv(tensors['inception_5b_pool'], 128, 1, 1, padding='SAME', biased=False, relu=False, name='inception_5b_pool_proj')
    tensors['inception_5b_pool_proj_bn'] = batch_norm(tensors['inception_5b_pool_proj'], is_training=is_training, relu=True, name='inception_5b_pool_proj_bn')

    tensors['inception_5b_output'] = concat(tensors['inception_5b_1x1_bn'], tensors['inception_5b_3x3_bn'], tensors['inception_5b_double3x3b_bn'], tensors['inception_5b_pool_proj_bn'], 3, name='inception_5b_output')
    tensors['Output'] = conv(tensors['inception_5b_output'], 2, 1, 1, padding='SAME', relu=False, name='Output')
    
    # Two deconvolution layers
    tensors['Outputx4'] = upsample(tensors['Output'], factor = 4, name = 'Outputx4')
    tensors['Outputx16'] = upsample(tensors['Outputx4'], factor = 4, name = 'Outputx16')

    # TODO
    tensors['Concat16'] = concat(tensors['Outputx16'], tensors['conv1_7x7_s2_bn_p'], 3, name='Concat16')  # TODO
    
    tensors['Convolution34'] = conv(tensors['Concat16'], 32, 1, 1, padding='SAME', biased=False, relu=False, name='Convolution34')
    tensors['BatchNorm34'] = batch_norm(tensors['Convolution34'], is_training=is_training, relu=True, name='BatchNorm34')
    tensors['Convolution35'] = conv(tensors['BatchNorm34'], 32, 3, 1, padding='SAME', biased=False, relu=False, name='Convolution35')
    tensors['BatchNorm35'] = batch_norm(tensors['Convolution35'], is_training=is_training, relu=True, name='BatchNorm35')

    tensors['Concat17'] = concat(tensors['Concat16'], tensors['BatchNorm35'], 3, name='Concat17')
    tensors['Convolution36'] = conv(tensors['Concat17'], 32, 1, 1, padding='SAME', biased=False, relu=False, name='Convolution36')
    tensors['BatchNorm36'] = batch_norm(tensors['Convolution36'], is_training=is_training, relu=True, name='BatchNorm36')
    tensors['Convolution37'] = conv(tensors['BatchNorm36'], 32, 3, 1, padding=None, biased=False, relu=False, name='Convolution37', dilation =2)
    tensors['BatchNorm37'] = batch_norm(tensors['Convolution37'], is_training=is_training, relu=True, name='BatchNorm37')

    tensors['Concat18'] = concat(tensors['Concat17'], tensors['BatchNorm37'], 3, name='Concat18')
    tensors['Convolution38'] = conv(tensors['Concat18'], 32, 1, 1, padding='SAME', biased=False, relu=False, name='Convolution38')
    tensors['BatchNorm38'] = batch_norm(tensors['Convolution38'], is_training=is_training, relu=True, name='BatchNorm38')
    tensors['Convolution39'] = conv(tensors['BatchNorm38'], 32, 3, 1, padding=None, biased=False, relu=False, name='Convolution39', dilation = 4)
    tensors['BatchNorm39'] = batch_norm(tensors['Convolution39'], is_training=is_training, relu=True, name='BatchNorm39')

    tensors['Concat19'] = concat(tensors['Concat18'], tensors['BatchNorm39'], 3, name='Concat19')
    tensors['Output_new_full1'] = conv(tensors['Concat19'], 1, 1, 1, padding='SAME', relu=False, name='Output_new_full1')

    # tensors['prob'] = softmax(tensors['Output_new_full1'], name='prob')
    tensors['logits'] = tensors['Output_new_full1']

    return tensors['logits']
