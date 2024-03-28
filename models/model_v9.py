# -*- coding:utf-8 -*-
# Time : 2022/10/30 20:51
# Author: haifwu
# File : model_v1.py

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class TaskNetwork(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super(TaskNetwork, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU()

        # encoding layers
        self.conv1 = ConvLayer(in_ch, 32, kernel_size=3, stride=1)
        self.in1_e = nn.BatchNorm2d(32)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.BatchNorm2d(64)

        self.conv3 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.in3_e = nn.BatchNorm2d(64)

        # residual layers
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(64, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.BatchNorm2d(64)

        self.deconv2 = UpsampleConvLayer(64, 64, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.BatchNorm2d(64)

        self.deconv1 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.in1_d = nn.BatchNorm2d(32)

        self.out = ConvLayer(32, out_ch, kernel_size=3, stride=1)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = self.relu(self.in1_d(self.deconv1(y)))
        y = self.out(y)

        return y


class RNet(nn.Module):
    def __init__(self, in_ch=1):
        super(RNet, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU()

        # encoding layers
        self.conv1 = ConvLayer(in_ch, 64, kernel_size=3, stride=1)
        self.in1_e = nn.BatchNorm2d(64)

        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.in2_e = nn.BatchNorm2d(64)

        self.conv3 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.in3_e = nn.BatchNorm2d(64)

        # residual layers
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)

        # decoding layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

        self.out = nn.Sigmoid()

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)

        y = self.pool(y)

        y = y.view(y.size(0), -1)

        # decode
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        y = self.out(y)

        return y

    # Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReplicationPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReplicationPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3

# Test model
if __name__ == '__main__':
    img = torch.rand([10, 1, 256, 256])

    output = RNet(in_ch=1, out_ch=1)(img)
    # print()
    print(output)




