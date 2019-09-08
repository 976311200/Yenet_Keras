import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import *
from keras import Model
# 定义SRM_lAYER
import numpy as np

# SRM_kernel shape = 5 , 5 , 1, 30

SRM_kernel = np.load('./SRM_Kernels.npy')


class SRM_layer(Layer):
    def __init__(self, output_dim, SRM_kernel, **kwargs):
        self.output_dim = output_dim
        self.SRM_kernel = SRM_kernel
        self.filter_num = (SRM_kernel.shape[3], SRM_kernel.shape[3])
        self.strides = (1, 1)
        super(SRM_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 这个函数是写需要更新的参数

        self.bias = self.add_weight(name='b', shape=[30], dtype=tf.float32, initializer=tf.constant_initializer(0.))


    def call(self, inputs, **kwargs):
        conv = K.conv2d(inputs, self.SRM_kernel, strides=(1, 1), padding='valid')
        return K.bias_add(conv, self.bias)
        # return conv

    def compute_output_shape(self, input_shape):
        # 这个是返回计算outputshape
        # 具体卷积计算公式我不是很清楚
        # 但根据Yenet输出是这样写的

        return (None, 252, 252, 30)
