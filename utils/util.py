# -*- coding:utf-8 -*-
# Time : 2022/10/30 17:05
# Author: haifwu
# File : util.py
import glob
import os
import re
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms


def findLastCheckpoint(save_dir, modelType):
    # Find from the path of the model
    file_list = glob.glob(os.path.join(save_dir, str(modelType) + '_*.pkl'))
    # If file_list is not empty, it means there is a trained model. Find the latest model.
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*" + str(modelType) + "_(.*).pkl.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    # If file_list is empty, it means there is no model. Let the initial epoch be 0, that is, start training from the 0th one.
    else:
        initial_epoch = 0
    return initial_epoch



class FFT_Guass_train_v6():
    def __init__(self, R_Scale):
        super(FFT_Guass_train_v6, self).__init__()
        self.R_Scale = R_Scale

    def fft(self, img):
        """
        Fourier Transform
        :param img: cv2.imread(path)
        :return: fft
        """
        # Forward fourier transform
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # Move the low frequency spectrum from the upper left corner to the center
        dft_shift = np.fft.fftshift(dft)

        return dft_shift

    def ifft(self, fshift):
        """
        Inverse Fourier Transform
        :param fshift: fft
        :return: iimg
        """
        # Move the low frequencies back to the upper left corner
        ishift = np.fft.ifftshift(fshift)
        # Inverse fourier transform
        iimg = cv2.idft(ishift)  # 逆变换
        iimg = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

        return iimg

    def guassian_kernel(self, height, width, sigma):
        sigma = sigma * self.R_Scale
        [X, Y] = np.meshgrid(np.arange(-height // 2, height // 2), np.arange(-width // 2, width // 2), indexing='ij')
        kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * (sigma ** 2) + 1e-5))
        # kernel = kernel / np.sum(kernel)

        return kernel

    def filter_high_pass_guassian(self, img, fshift, radius_ratio):
        """
        high-pass-filter
        removes low-frequency information in the central region
        :param img: image
        :param fshift: fft
        :param radius_ratio: filter radius
        :return:
        """
        # ------------------------------------------------------------------------------------------------- #
        #  Generate a circular filter, the value of the circle is 0, the rest of the filter is 1, filter
        # ------------------------------------------------------------------------------------------------- #
        # height, width
        height, width = img.shape
        mask = self.guassian_kernel(height, width, radius_ratio)
        mask = np.expand_dims(mask, axis=2)
        mask = np.concatenate((mask, mask), axis=2)
        low_frequency = mask * fshift
        return fshift - low_frequency, low_frequency, mask

    def imgifft(self, img, radius_ratio):
        # 1. fft
        fft = self.fft(img)
        result = fft.copy()
        # 2. get high_frequence
        high_frequency, low_frequncy, mask = self.filter_high_pass_guassian(img, fft.copy(), radius_ratio=radius_ratio)
        # 3. inverse fft
        ifft = self.ifft(high_frequency)
        # cv2 to PIL format
        ifft = (ifft - np.amin(ifft) + 0.00001) / (
                np.amax(ifft) - np.amin(ifft) + 0.00001)
        img = np.array(ifft * 255, np.uint8)
        img = Image.fromarray(img)

        # 打印高低频滤波后的频谱图
        result_h = 15 * np.log(cv2.magnitude(high_frequency[:, :, 0], high_frequency[:, :, 1]))
        result_l = 15 * np.log(cv2.magnitude(low_frequncy[:, :, 0], low_frequncy[:, :, 1]))
        result = 15 * np.log(cv2.magnitude(result[:, :, 0], result[:, :, 1]))

        result_h = np.array(result_h)
        result_l = np.array(result_l)
        result = np.array(result)

        # result_h.clip(0, 255)
        # result_l.clip(0, 255)
        result_h = Image.fromarray(result_h.astype(np.uint8))
        result_l = Image.fromarray(result_l.astype(np.uint8))
        result = Image.fromarray(result.astype(np.uint8))

        low_filter = mask[:, :, 0].squeeze()
        low_filter = np.array(low_filter * 255)
        low_filter.clip(0, 255)
        low_filter = Image.fromarray(low_filter.astype(np.uint8))

        return img, result_h, result_l, result, low_filter

    def tensorifft(self, img_tensor, radius_ratio):
        temp = torch.ones(img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3])
        target_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for i in range(img_tensor.shape[0]):
            current = img_tensor[i].cpu()
            img = current.squeeze().detach().numpy()
            ifft, high_frequency, low_frequency, complete_frequency, low_filter = self.imgifft(img, radius_ratio[i].cpu().detach().numpy())
            temp[i] = target_transform(ifft).unsqueeze(0)

        return temp, high_frequency, low_frequency, complete_frequency, low_filter

    def tensorfft(self, img_tensor):
        temp = torch.ones(img_tensor.shape[0], 2, img_tensor.shape[2], img_tensor.shape[3])
        target_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for i in range(img_tensor.shape[0]):
            current = img_tensor[i].cpu()
            img = current.squeeze().numpy()
            ifft = self.fft(img)
            ifft1 = Image.fromarray(ifft[:, :, 0])
            ifft2 = Image.fromarray(ifft[:, :, 1])
            tfft = torch.cat((target_transform(ifft1).unsqueeze(0), target_transform(ifft2).unsqueeze(0)), dim=1)
            temp[i] = tfft

        return temp


class FFT_Guass_test_v6():
    def __init__(self, R_Scale):
        super(FFT_Guass_test_v6, self).__init__()
        self.R_Scale = R_Scale

    def fft(self, img):
        """
        Fourier Transform
        :param img: cv2.imread(path)
        :return: fft
        """
        # Forward fourier transform
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # Move the low frequency spectrum from the upper left corner to the center
        dft_shift = np.fft.fftshift(dft)

        return dft_shift

    def ifft(self, fshift):
        """
        Inverse Fourier Transform
        :param fshift: fft
        :return: iimg
        """
        # Move the low frequencies back to the upper left corner
        ishift = np.fft.ifftshift(fshift)
        # Inverse fourier transform
        iimg = cv2.idft(ishift)  # 逆变换
        iimg = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

        return iimg

    def guassian_kernel(self, height, width, sigma):
        sigma = sigma * self.R_Scale
        [X, Y] = np.meshgrid(np.arange(-height // 2, height // 2), np.arange(-width // 2, width // 2), indexing='ij')
        kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * (sigma ** 2) + 1e-5))
        # kernel = kernel / np.sum(kernel)

        return kernel

    def filter_high_pass_guassian(self, img, fshift, radius_ratio):
        """
        high-pass-filter
        removes low-frequency information in the central region
        :param img: image
        :param fshift: fft
        :param radius_ratio: filter radius
        :return:
        """
        # ------------------------------------------------------------------------------------------------- #
        #  Generate a circular filter, the value of the circle is 0, the rest of the filter is 1, filter
        # ------------------------------------------------------------------------------------------------- #
        # height, width
        height, width = img.shape
        mask = self.guassian_kernel(height, width, radius_ratio)
        mask = np.expand_dims(mask, axis=2)
        mask = np.concatenate((mask, mask), axis=2)
        low_frequency = mask * fshift
        return fshift - low_frequency, low_frequency, mask

    def imgifft(self, img, radius_ratio):
        # 1. fft
        fft = self.fft(img * 255)
        result = fft.copy()
        # 2. get high_frequence
        high_frequency, low_frequncy, mask = self.filter_high_pass_guassian(img, fft.copy(), radius_ratio=radius_ratio)
        # 3. inverse fft
        ifft = self.ifft(high_frequency)
        # cv2 to PIL format
        ifft = (ifft - np.amin(ifft) + 0.00001) / (
                np.amax(ifft) - np.amin(ifft) + 0.00001)
        img = np.array(ifft * 255, np.uint8)
        img = Image.fromarray(img)

        # Print the spectrum diagram after high and low frequency filtering
        result_h = 15 * np.log(cv2.magnitude(high_frequency[:, :, 0], high_frequency[:, :, 1]))
        result_l = 15 * np.log(cv2.magnitude(low_frequncy[:, :, 0], low_frequncy[:, :, 1]))
        result = 15 * np.log(cv2.magnitude(result[:, :, 0], result[:, :, 1]))

        result_h = np.array(result_h)
        result_l = np.array(result_l)
        result = np.array(result)

        # result_h.clip(0, 255)
        # result_l.clip(0, 255)
        result_h = Image.fromarray(result_h.astype(np.uint8))
        result_l = Image.fromarray(result_l.astype(np.uint8))
        result = Image.fromarray(result.astype(np.uint8))

        low_filter = mask[:, :, 0].squeeze()
        low_filter = np.array(low_filter * 255)
        low_filter.clip(0, 255)
        low_filter = Image.fromarray(low_filter.astype(np.uint8))

        return img, result_h, result_l, result, low_filter

    def tensorifft(self, img_tensor, radius_ratio):
        temp = torch.ones(img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3])
        target_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for i in range(img_tensor.shape[0]):
            current = img_tensor[i].cpu()
            img = current.squeeze().detach().numpy()
            ifft, high_frequency, low_frequency, complete_frequency, low_filter = self.imgifft(img, radius_ratio[i].cpu().detach().numpy())
            temp[i] = target_transform(ifft).unsqueeze(0)

        return temp, high_frequency, low_frequency, complete_frequency, low_filter

    def tensorfft(self, img_tensor):
        temp = torch.ones(img_tensor.shape[0], 2, img_tensor.shape[2], img_tensor.shape[3])
        target_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for i in range(img_tensor.shape[0]):
            current = img_tensor[i].cpu()
            img = current.squeeze().numpy()
            ifft = self.fft(img)
            ifft1 = Image.fromarray(ifft[:, :, 0])
            ifft2 = Image.fromarray(ifft[:, :, 1])
            tfft = torch.cat((target_transform(ifft1).unsqueeze(0), target_transform(ifft2).unsqueeze(0)), dim=1)
            temp[i] = tfft

        return temp



def guassian_kernel(height, width, sigma):
    [X, Y] = np.meshgrid(np.arange(-height // 2, height // 2), np.arange(-width // 2, width // 2), indexing='ij')
    kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * (sigma ** 2) + 1e-10))
    # kernel = kernel / np.sum(kernel)

    return kernel


def guassian_nonormal_kernel(height, width, sigma):
    sigma = int(sigma * min(height, width) // 2)
    [X, Y] = np.meshgrid(np.arange(-height // 2, height // 2), np.arange(-width // 2, width // 2), indexing='ij')
    kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * (sigma ** 2) + 1e-10))
    # kernel = kernel / np.sum(kernel)

    return kernel

def guassian_nonormal_rand_kernel(height, width, sigma):
    sigma = sigma * min(height, width) // 2
    [X, Y] = np.meshgrid(np.arange(-height // 2, height // 2), np.arange(-width // 2, width // 2), indexing='ij')
    kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * (sigma ** 2) + 1e-10))
    # kernel = kernel / np.sum(kernel)

    return kernel


def guassian_normal_kernel(height, width, sigma):
    sigma = int(sigma * min(height, width) // 2)
    [X, Y] = np.meshgrid(np.arange(-height // 2, height // 2), np.arange(-width // 2, width // 2), indexing='ij')
    kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * (sigma ** 2) + 1e-10))
    kernel = kernel / np.sum(kernel)

    return kernel
    

def guassian_mean_kernel(height, width, sigma):
    mask = np.zeros((height, width))
    height_start = int((height * sigma) // 2)
    width_start = int((width * sigma) // 2)
    sigma = int(sigma * min(height, width) // 2)
    [X, Y] = np.meshgrid(np.arange(-height_start, height_start), np.arange(-width_start, width_start), indexing='ij')

    kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * (sigma ** 2) + 1e-10))
    height_left = int(height // 2 - height_start)
    width_left = int(width // 2 - width_start)
    mask[height_left: height_left + height_start * 2, width_left: width_left + width_start * 2] = kernel

    return 1 - mask


def guassian_mean_normal_kernel(height, width, sigma):
    mask = np.zeros((height, width))
    height_start = int((height * sigma) // 2)
    width_start = int((width * sigma) // 2)
    sigma = int(sigma * min(height, width) // 2)
    [X, Y] = np.meshgrid(np.arange(-height_start, height_start), np.arange(-width_start, width_start), indexing='ij')
    kernel = np.exp(- (X ** 2 + Y ** 2) / (2 * np.pi * (sigma ** 2) + 1e-10))
    kernel = kernel / np.sum(kernel)
    height_left = int(height // 2 - height_start)
    width_left = int(width // 2 - width_start)
    mask[height_left: height_left + height_start * 2, width_left: width_left + width_start * 2] = kernel

    return 1 - mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)






