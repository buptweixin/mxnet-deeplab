#!/usr/bin/env python
# encoding: utf-8

import mxnet as mx

# 生成DeepLab v2 网络文件
# VGG 16部分
def vgg_16(input, workspace_default=1024):
    conv1_1 = mx.sym.Convolution(data=input, kernel=(3, 3), \
                                 pad=(1, 1), num_filter=64, \
                                 workspace=workspace_default, \
                                 attr={'weight_lr_mult':'1', 'bias_lr_mult': '2'},
                                 name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", \
                                name = "relu1_1")
    conv1_2 = mx.sym.Convolution(data=relu1_1, \
                                kernel=(3, 3), pad=(1, 1), num_filter=64,
                                workspace=workspace_default, name="conv1_2")
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu", \
                               name="relu1_2")
    pool1 = mx.sym.Pooling(data=relu1_2, pool_type="max", kernel=(3, 3), \
                           stride=(2, 2), pad=(1, 1), name="pool1")

    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), \
                                num_filter=128, \
                                 workspace=workspace_default, \
                                 name="conv2_1")
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", \
                                name="relu2_1")
    conv2_2 = mx.sym.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), \
                                num_filter=128, \
                                workspace=workspace_default, \
                                 name="conv2_2")
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu", \
                                name="relu2_2")
    pool2 = mx.sym.Pooling(data=relu2_2, pool_type="max", kernel=(3, 3), \
                           pad=(1, 1), stride=(2, 2), name="pool2")

    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), \
                                num_filter=256, name="conv3_1", \
                                workspace=workspace_default)
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", \
                               name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), \
                                num_filter=256, name="conv3_2", \
                                workspace=workspace_default)
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", \
                                name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), \
                                num_filter=256, name="conv3_3", \
                                workspace=workspace_default)
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", \
                                 name="relu3_3")
    pool3 = mx.sym.Pooling(data=relu3_3, kernel=(3, 3), stride=(2, 2), \
                          pad=(1, 1), name="pool3", pool_type="max")

    conv4_1 = mx.sym.Convolution(data=pool3, pad=(1, 1), num_filter=512, \
                                kernel=(3, 3), name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", \
                               name="relu4_1")
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), \
                                num_filter=512, name="conv4_2", \
                                workspace=workspace_default)
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", \
                               name="relu4_2")
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), \
                               num_filter=512, name="conv4_3", \
                                workspace=workspace_default)
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu", \
                               name="relu4_3")

    pool4 = mx.sym.Pooling(data=relu4_3, pool_type="max", kernel=(3, 3), \
                          pad=(1, 1), stride=(1, 1), name="pool4")
    conv5_1 = mx.sym.Convolution(data=pool4, kernel=(3, 3), dilate=(2, 2),
                                pad=(2, 2), num_filter=512, name="conv5_1", \
                                workspace=workspace_default)
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", \
                                name="relu5_1")
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(3, 3), dilate=(2, 2),
                                pad=(2, 2), num_filter=512, name="conv5_2", \
                                workspace=workspace_default)
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", \
                               name="relu5_2")
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3, 3), dilate=(2, 2),
                                pad=(2, 2), num_filter=512, name="conv5_3", \
                                workspace=workspace_default)
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu", \
                               name="relu5_3")
    pool5 = mx.sym.Pooling(data=relu5_3, pool_type="max", kernel=(3, 3), \
                          stride=(1, 1), pad=(1, 1), name="pool5")
    pool5a = mx.sym.Pooling(data=pool5, pool_type="avg", kernel=(3, 3), \
                             stride=(1, 1), pad=(1, 1), name="pool5a")
    return pool5a

# FC8
def fully_convolution(input, workspace_default=1024, numclass=6):
    fc6 = mx.sym.Convolution(data=input, num_filter=1024, pad=(12, 12), \
                            dilate=(12, 12), kernel=(3, 3), name="fc6", \
                            workspace=workspace_default)
    relu6 = mx.sym.Activation(data=fc6, act_type="relu", \
                             name="relu6")
    drop6 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.sym.Convolution(data=drop6, kernel=(1, 1), num_filter=1024, \
                             workspace=workspace_default, name="fc7")
    relu7 = mx.sym.Activation(data=fc7, act_type="relu", \
                             name="relu7")
    drop7 = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")
    fc8_exper = mx.sym.Convolution(data=drop7, num_filter=numclass, kernel=(1, 1), \
                                  name="fc8_exper", \
                                  workspace=workspace_default)
    return fc8_exper

# SoftmaxOutput
def softmax_output(input):
    softmax = mx.sym.SoftmaxOutput(data=input, multi_output=True, \
                                name="softmax")
    return softmax

# 生成训练网络
def get_deeplab_symbol_train(workspace_default=1536, numclass=6):
    data = mx.sym.Variable('data')
    pool5 = vgg_16(data, workspace_default)
    fc8 = fully_convolution(pool5, workspace_default, numclass)
    softmax = softmax_output(fc8)
    return softmax

# 生成测试网络
def get_deeplab_symbol_test(workspace_default=1536, numclass=6):
    data = mx.sym.Variable('data')
    pool5 = vgg_16(data, workspace_default)
    fc8 = fully_convolution(pool5, workspace_default, numclass)
#     upsamping = mx.sym.UpSampling(
#         fc8, scale=8, numclass, sample_type='bilinear', num_args = 2, name="upsampling")
    return fc8

