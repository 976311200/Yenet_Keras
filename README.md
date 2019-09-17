# Yenet_Keras
Yenet Code by Keras

## 使用方法

在test_data放validation的数据

在train_data里的stego和cover分别放入图片

主程序是My_Yenet_Keras

一些操作函数实现是在ops.py

SRM算子在SRM_layer.py

TLU截断层在TLU.py

## 测试
测试程序是Yenet_test
会按照内部定义的路径，将预测labels保存至txt文件当中

## 继续训练
使用Yenet_Continue.py
设置好模型路径后
即可导入原有模型继续训练
