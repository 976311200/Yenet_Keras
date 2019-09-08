import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import *
from keras import Model


class TLU_layer(Layer):
    """
    定义TLU激活层
    """
    def __init__(self, output_dim, TLU_Threshold, **kwargs):
        self.output_dim = output_dim
        self.TLU_Threshold = TLU_Threshold
        super(TLU_layer, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        return tf.clip_by_value(inputs, -self.TLU_Threshold, self.TLU_Threshold, name='TLU')

    def compute_output_shape(self, input_shape):
        # 这个是返回计算outputshape

        return input_shape

# just test

# inputs = Input(shape=(256, 256, 30))
# print(type(inputs))
# conv1 = Conv2D(filters=30, kernel_size=(5, 5), padding='VALID', name='Group1Conv1')(inputs)
# TLU = TLU_layer(output_dim=(252, 252, 30), TLU_Threshold=3)(conv1)
# print(type(TLU))
# print(TLU.shape)
# model = Model(input=inputs, output=TLU)
# print(model.summary())
