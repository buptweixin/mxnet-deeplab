#!/usr/bin/env python
# encoding: utf-8

import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from PIL import Image
import math
import cv2
#####################
#定义数据预期类
####################

class FileIter(DataIter):
    def __init__(self, root_dir, flist_name,
                 shrink = 0,
                 # root_dir: 包含图片文件夹的根地址
                 # flist_name: 包含数据和对应标签图像路径的txt文件，以`\t`分隔
                 # data_name: 数据在网络中的名称
                 # label_name: 标签在网络中的名称
                rgb_mean = (117, 117, 117),
                cut_off_size = None,
                data_name = "data",
                label_name = "softmax_label"):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.shrink = float(shrink)
        self.mean = np.array(rgb_mean)
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name

        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.f = open(self.flist_name, 'r')
        self.data, self.label = self._read()
        self.cursor = -1

    def _read(self):
        # 读入一张图片
        data_img_name, label_img_name = self.f.readline().strip('\n').split("\t")
        data = {}
        label = {}
        data[self.data_name], label[self.label_name] = self._read_img(data_img_name, label_img_name)
        return list(data.items()), list(label.items())

    def _read_img(self, img_name, label_name):
        # Image库读取图像通道axis为(height, width, channel)
        img = Image.open(os.path.join(self.root_dir, img_name))  # (h, w, c)
        label = Image.open(os.path.join(self.root_dir, label_name))

        assert img.size == label.size
        img = np.array(img, np.float32)
        label = np.array(label)
        if self.cut_off_size is not None:
            #判断是否需要裁切图片
            max_hw = max(img.shape[0], img.shape[1])
            min_hw = min(img.shape[0], img.shape[1])
            if min_hw > self.cut_off_size:
                rand_start_max = round(np.random.uniform(0, max_hw - self.cut_off_size - 1))
                rand_start_min = round(np.random.uniform(0, min_hw - self.cut_off_size - 1))
                if img.shape[0] == max_hw:
                    img = img[rand_start_max : rand_start_max + self.cut_off_size, \
                              rand_start_min : rand_start_min + self.cut_off_size]
                    label = label[rand_start_max : rand_start_max + self.cut_off_size, \
                                  rand_start_min : rand_start_min + self.cut_off_size]
                else:
                    img = img[rand_start_min : rand_start_min + self.cut_off_size, \
                              rand_start_max : rand_start_max + self.cut_off_size]
                    label = label[rand_start_min : rand_start_min + self.cut_off_size, \
                              rand_start_max : rand_start_max + self.cut_off_size]
            elif max_hw > self.cut_off_size:
                rand_start = round(np.random.uniform(0, max_hw - min_hw - 1))
                if img.shape[0] == max_hw:
                    img = img[rand_start : rand_start + min_hw, :]
                    label = label[:, rand_start : rand_start + min_hw]
        # 图片减去均值
        reshaped_mean = self.mean.reshape(1, 1, 3)
        img = img - reshaped_mean
        # 将图像axis由(h, w, c) 转换为(c, h, w)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2) # (c, h, w)
        img = np.expand_dims(img, axis=0) # (1, c, h, w)
        label = np.array(label)
        # 判断是否需要进行缩放
        if self.shrink > 0:
            label = cv2.resize(label, (int(math.ceil(len(label[0]) / self.shrink)), \
                                       int(math.ceil(len(label) / self.shrink))))
        label = np.expand_dims(label, axis=0) # (1, h, w)

        return (img, label)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]

    def get_batch_size(self):
        return 1

    def reset(self):
        self.cursor = -1
        self.f.close()
        self.f = open(self.flist_name, 'r')

    def iter_next(self):
        self.cursor += 1
        if (self.cursor < self.num_data - 1):
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            self.data, self.label = self._read()
            return {self.data_name : self.data[0][1],
                   self.label_name : self.label[0][1]}
        else:
            raise StopIteration

