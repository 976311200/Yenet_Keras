from PIL import Image
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
import glob
import pickle
import numpy as np
import tensorflow.gfile as gfile
import tensorflow as tf
from ops import rgb2gray, Data_Process_For_TestValidation, Data_Process_For_Train, validation, Normalize,fit_keras_channels
from SRM_Layer import SRM_layer
from TLU import TLU_layer

BATCH_SIZE = 20 # 批处理图片个数 默认为50 前期测试
EPOCHS = 1 # DEFAULT 10
IMAGE_SIZE = 256 # 输入的是256 x 256
NUM_CHANNEL = 1 # 通道为1，默认是灰度图
NUM_LABELS = 2 # Labels个数，有隐写或无隐写
TLU_Threshold = 3 # 定义TLU截断算子阈值

OPT = 'Nadam' # 原来Xu-net使用的是momentum优化器，Keras没有，所以选了个比较接近的Nadam
LOSS = 'categorical_crossentropy' # 二分类问题，这里我选用的是交叉熵作为损失函数

#BN_DECAY = 0.95
#UPDATE_OPS_COLLECTION = 'Discriminative_update_ops'
filename_str = '{}ye_net_{}_{}_bs_{}_epochs_{}{}'
MODEL_DIR = './model/train_demo/'
MODEL_FORMAT = '.h5'
HISTORY_DIR = './history/train_continued/'
HISTORY_FORMAT = '.history'

# 模型文件
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
MODEL_NAME = './ye_net_Nadam_categorical_crossentropy_bs_25_epochs_2000.h5'

# 训练记录文件
HISTORY_FILE = filename_str.format(HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)

# validation_data.txt文件路径
TXT_DIR = r'./valid_labels.txt'

TRAIN_DATA_DIR = './train_data' # 训练数据的路径
TEST_DATA_DIR = './test_data' # 测试数据的路径

SRM_kernel = np.load('./SRM_Kernels.npy')

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


"""
将train目录下图片输入给X_train Y_train
"""
X_train, Y_train = Data_Process_For_Train(TRAIN_DATA_DIR)
# print(one_hot)

"""
将验证集数据输入给X_test, Y_test
"""
X_test, Y_test = Data_Process_For_TestValidation(TEST_DATA_DIR, TXT_DIR)

X_train = Normalize(X_train)
# Fit Keras Channel

X_train, input_shape = fit_keras_channels(X_train, rows=IMAGE_SIZE, cols=IMAGE_SIZE)

model = load_model(MODEL_DIR + MODEL_NAME, custom_objects={'SRM_layer': SRM_layer, 'TLU_layer':TLU_layer})

# 查看模型摘要
print(model.summary())

# 模型可视化
# Keras的可视化使用的是utils下的plot model
# plot_model(model, to_file=MODEL_VIS_FILE, show_shapes=True)

# just test
# print("wait")
# print(X_test.shape)
# print(Y_test.shape)

# 继续训练模型
history= model.fit(X_train,
          Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2,
          validation_data=(X_test, Y_test))


# 保存模型
if not gfile.Exists(MODEL_DIR):
    gfile.MakeDirs(MODEL_DIR)

model.save(MODEL_FILE)
print('Saved trained model at %s ' % MODEL_FILE)