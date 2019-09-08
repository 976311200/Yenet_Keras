from PIL import Image
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
import glob
import pickle
import numpy as np
import tensorflow.gfile as gfile
import matplotlib.pyplot as plt
from scipy import ndimage

def resize(img, width=256, height=256):
    """
    将1024 1024图像缩放至256 256
    :param img:
    :return:缩放后的图像
    """
    img_resized = img.resize((width, height), Image.ANTIALIAS)

    return img_resized

# 转换为灰度图
def rgb2gray(img):
    """
    利用公式对灰度图进行转换
    :param img:
    :return:
    """
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

# X_train = rgb2gray(X_train)


def Data_Process_For_Train(DATA_DIR):
    """
    分别将训练数据读入X_train中
    stego和cover图像一半一半
    前50为stego
    后50为cover
    返回X_train 和经过one_hot编码的 100 x 2数组
    :param TRAIN_DATA_DIR:
    :return: X_train np.array类型
             one_hot编码 length， 2 的向量
    """
    X_train = []

    # 先添加的是stego
    for filename in glob.glob(DATA_DIR + '/stego/' + '*.jpg'):
        # print(filename)

        # 先做一次resize缩放
        img = resize(Image.open(filename))
        img = np.array(img) # 将缩放后的图片转换成np array
        img = rgb2gray(img).reshape(img.shape[0], img.shape[1], 1)  # 先将图片转换成灰度图
        # 做高通滤波
        # highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0], img.shape[1], 1)
        X_train.append(img) # 将滤波后的图片添加进X_train

    # 后添加的是cover
    for filename in glob.glob(DATA_DIR + '/cover/' + '*.jpg'):
        # print(filename)
        img = resize(Image.open(filename))
        img = np.array(img)  # 将缩放后的图片转换成np array
        img = rgb2gray(img).reshape(img.shape[0], img.shape[1], 1)  # 先将图片转换成灰度图

        # 做高通滤波
        # highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0], img.shape[1], 1)
        X_train.append(img)

    length = len(X_train)
    # 转换为np.array类型
    X_train = np.array(X_train, dtype=np.float32)
    one_hot = np.zeros((length, 2))
    # print(one_hot.shape)
    half_length = length // 2
    for i in range(half_length):
        one_hot[i, 0] = 1
        one_hot[half_length + i, 1] = 1

    # [1, 0]的是stego [0， 1]的是cover


    # print(one_hot)
    return X_train, one_hot

def validation(TXT_DIR, length):
    """
    用于读取validation_data
    :param TXT_DIR: validationdata的路径
    :param length: 我们要读的长度，这里会将X_test的长度传入
    :return: np.array  validate  是一个length * 2 的向量
    """
    with open(TXT_DIR) as labels:
        # 使用splitlines去掉换行符\n
        # 使用read方法计数的时候，会把换行符给计入，所以我们要在原有的长度乘2
        a = labels.read(length*2).splitlines()
        # print(a)
        # print(len(a))
        # list 类型
        # print(type(a))
        # a = np.array(a)

    validate = []

    for i in a:
        # 由于读入的是字符串，所以判定不能直接拿数字0，而是字符0
        if i == '0':
            validate.append([0, 1])
        else :
            validate.append([1, 0])

    # print(validate)
    validate = np.array(validate)
    # print(validate.shape)
    return validate


def Data_Process_For_TestValidation(DATA_DIR, TXT_DIR):
    """
    处理验证的图片数据
    :param DATA_DIR: 训练集的路径
    TXT_DIR: 验证集标签txt文件的路径
    :return:
    """
    X_test = []

    # 先添加的是stego
    for filename in glob.glob(DATA_DIR + '/*.jpg'):
        # print(filename)
        # 先做一次resize缩放
        img = resize(Image.open(filename))
        img = np.array(img)  # 将缩放后的图片转换成np array
        img = rgb2gray(img).reshape(img.shape[0], img.shape[1], 1)  # 先将图片转换成灰度图
        # 做高通滤波
        # highpassed_img = ndimage.convolve(img, kernel_5x5, mode='constant', cval=0.0).reshape(img.shape[0],
        #                                                                                       img.shape[1], 1)
        X_test.append(img) # 将滤波后的图片添加进X_train

    length = len(X_test) # 用于文档读取

    validate = validation(TXT_DIR, length)
    X_test = np.array(X_test, dtype=np.float32)

    return X_test, validate

def Normalize(X_train):
    """
    对数据规范化
    也就是将灰度值除以255
    :param X_train:
    :return:
    """
    return X_train / 255

def fit_keras_channels(batch, rows=1024, cols=1024):
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    return batch, input_shape