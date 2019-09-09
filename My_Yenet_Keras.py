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
HISTORY_DIR = './history/train_demo/'
HISTORY_FORMAT = '.history'

# 模型网络结构文件
MODEL_VIS_FILE = 'ye_net_classfication' + '.png'
# 模型文件
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
# 训练记录文件
HISTORY_FILE = filename_str.format(HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)

# validation_data.txt文件路径
TXT_DIR = r'./valid_labels.txt'

TRAIN_DATA_DIR = './train_data' # 训练数据的路径
TEST_DATA_DIR = './test_data' # 测试数据的路径

SRM_kernel = np.load('./SRM_Kernels.npy')

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


# Pre processing
inputs = Input(shape=input_shape, name="inputs")
SRM = SRM_layer(output_dim=(252, 252, 30), SRM_kernel=SRM_kernel, name='SRM_Layer')(inputs)
TLU = TLU_layer(output_dim=(252, 252, 30), TLU_Threshold=TLU_Threshold, name='TLU')(SRM)
# Group1
conv1 = Conv2D(filters=30, kernel_size=(5, 5), padding='SAME', name='Conv1')(TLU)
bn1 = BatchNormalization()(conv1)
relu1 = Activation('relu', name='relu1')(bn1)

# Group2
conv2 = Conv2D(filters=30, kernel_size=(3, 3), padding='VALID', name='Conv2')(relu1)
bn2 = BatchNormalization()(conv2)
relu2 = Activation('relu', name='relu2')(bn2)

# Group3
conv3 = Conv2D(filters=30, kernel_size=(3, 3), padding='VALID', name='Conv3')(relu2)
bn3 = BatchNormalization()(conv3)
relu3 = Activation('relu', name='relu3')(bn3)

# Group4
conv4 = Conv2D(filters=30, kernel_size=(3, 3), padding='VALID', name='Conv4')(relu3)
bn4 = BatchNormalization()(conv4)
relu4 = Activation('relu', name='relu4')(bn4)
pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME', name='AveragePool1')(relu4)

# Group5
conv5 = Conv2D(filters=32, kernel_size=(5, 5), padding='VALID', name='Conv5')(pool1)
bn5 = BatchNormalization()(conv5)
relu5 = Activation('relu', name='relu5')(bn5)
pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='VALID', name='AveragePool2')(relu5)

# Group6
conv6 = Conv2D(filters=32, kernel_size=(5, 5), padding='VALID', name='Conv6')(pool2)
bn6 = BatchNormalization()(conv6)
relu6 = Activation('relu', name='relu6')(bn6)
pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='VALID', name='AveragePool3')(relu6)

# Group7
conv7 = Conv2D(filters=32, kernel_size=(5, 5), padding='VALID', name='Conv7')(pool3)
bn7 = BatchNormalization()(conv7)
relu7 = Activation('relu', name='relu7')(bn7)
pool4 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='VALID', name='AveragePool4')(relu7)

# Layer8
conv8 = Conv2D(filters=16, kernel_size=(3, 3), padding='VALID', name='Conv8')(pool4)
bn8 = BatchNormalization()(conv8)
relu8 = Activation('relu', name='relu8')(bn8)

# Layer9
conv9 = Conv2D(filters=16, kernel_size=(3, 3), strides=(3, 3), padding='VALID', name='Conv9')(relu8)
bn9 = BatchNormalization()(conv9)
relu9 = Activation('relu', name='relu9')(bn9)

# FullConnected
full_connect = Dense(16, activation='relu', name='full_connected')(relu9)

# Flatten
flatten_layer =Flatten(name='Flatten_Layer')(full_connect)

# Softmax
softmax = Dense(NUM_LABELS, activation='softmax', name='Classification')(flatten_layer)


# 定义模型的输入与输出
model = Model(inputs=inputs, outputs=softmax)
model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'])

# 查看模型摘要
print(model.summary())

# 模型可视化
# Keras的可视化使用的是utils下的plot model
plot_model(model, to_file=MODEL_VIS_FILE, show_shapes=True)

# just test
# print("wait")
# print(X_test.shape)
# print(Y_test.shape)

# 开始训练模型
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