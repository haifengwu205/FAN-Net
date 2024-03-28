# -*- coding: utf-8 -*-
# @Time:   2020/3/5 18:22
# @Author: haifwu

# ===========================================================
# ===  Life is Short I Use Python!!!                      ===
# ===  If this runs wrong,don't ask me,I don't know why;  ===
# ===  If this runs right,thank god,and I don't know why. ===
# ===  Maybe the answer,my friend,is blowing in the wind. ===
# ===========================================================

# @Project : pytorch_code
# @FileName: datasetList.py
# @Software: PyCharm
# @Blog:     https://www.cnblogs.com/haifwu/

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
# import skimage.io as io
import csv
from PIL import Image
import cv2
from utils import util


class DatasetList(Dataset):
    # ==================================================
    # csv_path: the path of csv file
    # transform, target_transform: image, label transform
    # ==================================================
    def __init__(self, csv_path, input_transform=None, target_transform=None, gray_transform=None, co_transform=None, ratio=0.5):
        self.data_info = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.gray_transform = gray_transform
        self.co_transform = co_transform
        self.ratio = ratio

    def __getitem__(self, index):
        source_image1_path = self.data_info.iloc[index, 0]
        # source_image1 = Image.open(source_image1_path)
        source_image1 = cv2.imread(source_image1_path, 0)
        source_image1 = util.FFT().imgifft(source_image1, self.ratio)

        # second col
        source_image2_path = self.data_info.iloc[index, 1]
        # source_image2 = Image.open(source_image2_path)
        source_image2 = cv2.imread(source_image2_path, 0)
        source_image2 = util.FFT().imgifft(source_image2, self.ratio)

        source_image3_path = self.data_info.iloc[index, 3]
        source_image3 = Image.open(source_image3_path)
        # third col
        source_label_path = self.data_info.iloc[index, 2]
        source_label = Image.open(source_label_path)

        # target_image1_path = self.data_info.iloc[index, 4]
        # target_image1 = Image.open(target_image1_path)
        # # second col
        # target_image2_path = self.data_info.iloc[index, 5]
        # target_image2 = Image.open(target_image2_path)
        #
        # target_image3_path = self.data_info.iloc[index, 5]
        # target_image3 = Image.open(target_image3_path)

        if self.co_transform is not None:
            source_image1, source_image2, source_image3, source_label = self.co_transform(source_image1, source_image2, source_image3, source_label)

        # if self.co_transform is not None:
        #     target_image1, target_image2, target_image3, target_label = self.co_transform(target_image1, target_image2,
        #                                                                               target_image3, target_image3)
        # numpy to tensor
        if self.input_transform is not None:
            source_image1 = self.input_transform(source_image1)
            source_image2 = self.input_transform(source_image2)
            source_image3 = self.input_transform(source_image3)
            # target_image1 = self.input_transform(target_image1)
            # target_image2 = self.input_transform(target_image2)
            # target_image3 = self.input_transform(target_image3)

        if self.target_transform is not None:
            source_label = self.target_transform(source_label)

        if source_label.shape[0] == 3:
            source_new_label = source_label[0]
            source_new_label = source_new_label.unsqueeze(dim=0)

        else:
            source_new_label = source_label

        # image = torch.cat((img1, img2), dim=0)

        # one_hot for label
        label_one_hot = torch.zeros([2, source_label.shape[1], source_label.shape[2]])
        label_one_hot.scatter_(0, source_new_label.long(), 1)

        return source_image1, source_image2, source_image3, label_one_hot, source_new_label
        # return source_image1, source_image2, source_image3, label_one_hot, source_new_label, target_image1, target_image2, target_image3

    def __len__(self):
        return len(self.data_info)


def data_set(csv_path, input_transform=None, target_transform=None, gray_transform=None, co_transform=None, ratio=0.5):
    train_data = DatasetList(csv_path, input_transform, target_transform, gray_transform, co_transform, ratio)
    return train_data


if __name__ == "__main__":
    # data = torch.randn((32, 2, 32, 32))
    data = np.random.randn(32, 2, 32, 32)
    minvalue = np.percentile(data, 0)
    print(minvalue)