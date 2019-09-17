from PIL import Image
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
import glob
import pickle
import numpy as np
import tensorflow.gfile as gfile
from ops import rgb2gray, Data_Process_For_TestValidation, Data_Process_For_Train, validation, Normalize,fit_keras_channels
from ops import prediction_v2
from TLU import TLU_layer
import tensorflow as tf

filename_str = '{}ye_net_{}_{}_bs_{}_epochs_{}{}'
MODEL_DIR = './model/train_demo/'
MODEL_FORMAT = '.h5'
HISTORY_DIR = './history/train_demo/'
MODEL_NAME = './ye_net_Nadam_categorical_crossentropy_bs_25_epochs_2000.h5'
HISTORY_FORMAT = '.history'

TEST_DATA_DIR = 'D:/信息隐藏比赛/test(1)'  # 测试数据的路径
TXT_DIR = r'./valid_labels.txt'

# 保存labels的txt
PREDICT_LABELS_DIR = 'result_labels2.txt'

#
SRM_kernel = np.load('./SRM_Kernels.npy')
TLU_Threshold = 3 # 定义TLU截断算子阈值


# 给出SRM_layer定义
# 报错是因为要在init传入默认值
class SRM_layer(Layer):
    def __init__(self, output_dim=(252, 252, 30), SRM_kernel=SRM_kernel, **kwargs):
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


class TLU_layer(Layer):
    """
    定义TLU激活层
    """
    def __init__(self, output_dim=(252, 252, 30), TLU_Threshold=TLU_Threshold, **kwargs):
        self.output_dim = output_dim
        self.TLU_Threshold = TLU_Threshold
        super(TLU_layer, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        return tf.clip_by_value(inputs, -self.TLU_Threshold, self.TLU_Threshold, name='TLU')

    def compute_output_shape(self, input_shape):
        # 这个是返回计算outputshape

        return input_shape

# labels 前20
#  0 0 0 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 0 1

# 导入已有的模型
Yenet_model_pretrained = load_model(MODEL_DIR + MODEL_NAME, custom_objects={'SRM_layer': SRM_layer, 'TLU_layer':TLU_layer})

X_test, Y_test = Data_Process_For_TestValidation(TEST_DATA_DIR, TXT_DIR)

print("读入数据完毕，共需预测图像有{}张".format(X_test.shape[0]))

# 网络预测值
pretrained_predict = Yenet_model_pretrained.predict(X_test)
# 进行数值处理
labels = prediction_v2(pretrained_predict)

print(labels.shape)


np.savetxt(PREDICT_LABELS_DIR, labels, delimiter='\n', fmt='%d')

print(labels)