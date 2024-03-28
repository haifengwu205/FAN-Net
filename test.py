# -*- coding:utf-8 -*-
# Time : 2022/10/31 14:43
# Author: haifwu
# File : test.py

import argparse
import os

import cv2
import pywt
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import time
from utils import util, fusion_morphology


class Main:
    def __init__(self, opt):
        super(Main, self).__init__()
        self.file_ext = opt.test_file_ext
        self.img_root = opt.test_dataset_path
        self.dataset = opt.dataset
        self.save_root = opt.test_save_path
        self.model_R_dir = opt.test_model_R
        self.model_T_dir = opt.test_model_T
        self.device = torch.device("cuda:0" if opt.cuda else "cpu")
        # self.device = torch.device("cpu")
        self.opt = opt


    def funm(self):
        # ********************************************  load model ********************************************
        model_R = torch.load(self.model_R_dir, map_location=self.device)
        model_T = torch.load(self.model_T_dir, map_location=self.device)

        total_params = sum(p.numel() for p in model_R.parameters())
        print('RNet, Total parameters: %.6fM (%d)' % (total_params / 1e6, total_params))
        total_trainable_params = sum(
            p.numel() for p in model_R.parameters() if p.requires_grad)
        print('RNet, Total_trainable_params: %.6fM (%d)' % (total_trainable_params / 1e6, total_trainable_params))

        total_params = sum(p.numel() for p in model_T.parameters())
        print('DNet, Total parameters: %.6fM (%d)' % (total_params / 1e6, total_params))
        total_trainable_params = sum(
            p.numel() for p in model_T.parameters() if p.requires_grad)
        print('DNet, Total_trainable_params: %.6fM (%d)' % (total_trainable_params / 1e6, total_trainable_params))
        model_R.eval()
        model_T.eval()

        # ********************************************  parameters ********************************************
        for dataset in self.dataset:
            print(dataset)
            img1_root = os.path.join(self.img_root, dataset, 'img1')
            img2_root = os.path.join(self.img_root, dataset, 'img2')
            img1_root = img1_root.replace('\\', '/')
            img2_root = img2_root.replace('\\', '/')

            decisionMap_save_root = os.path.join(self.save_root, dataset, 'decisionmap')
            fusion_save_root = os.path.join(self.save_root, dataset, 'fusion')
            decisionMap_morphology_save_root = os.path.join(self.save_root, dataset, 'morphology')
            decisionMap_crf_save_root = os.path.join(self.save_root, dataset, 'decisionCrf')
            fusion_crf_save_root = os.path.join(self.save_root, dataset, 'crf')

            decisionMap_save_root = decisionMap_save_root.replace('\\', '/')
            fusion_save_root = fusion_save_root.replace('\\', '/')
            decisionMap_morphology_save_root = decisionMap_morphology_save_root.replace('\\', '/')
            decisionMap_crf_save_root = decisionMap_crf_save_root.replace('\\', '/')
            fusion_crf_save_root = fusion_crf_save_root.replace('\\', '/')


            if not os.path.exists(decisionMap_save_root):
                os.makedirs(decisionMap_save_root)
            if not os.path.exists(fusion_save_root):
                os.makedirs(fusion_save_root)
            if not os.path.exists(decisionMap_morphology_save_root):
                os.makedirs(decisionMap_morphology_save_root)
            if not os.path.exists(decisionMap_crf_save_root):
                os.makedirs(decisionMap_crf_save_root)
            if not os.path.exists(fusion_crf_save_root):
                os.makedirs(fusion_crf_save_root)

            img_names = os.listdir(img1_root)

            # ********************************************  process  ********************************************
            begain_time = time.time()
            with torch.no_grad():
                for img_name in img_names:
                    print(img_name)
                    name, suffix = img_name.split('.')
                    img1_dir = os.path.join(img1_root, img_name)
                    img2_dir = os.path.join(img2_root, img_name.replace('A', 'B'))


                    decisionMap_savePath = os.path.join(decisionMap_save_root, name.replace('-A', '') + '.png')
                    fusion_savePath = os.path.join(fusion_save_root, name.replace('-A', '') + '.png')\

                    decisionMap_morphology_savePath = os.path.join(decisionMap_morphology_save_root, name.replace('-A', '') + '.png')
                    decisionMap_crf_savePath = os.path.join(decisionMap_crf_save_root, name.replace('-A', '') + '.png')
                    fusion_crf_savePath = os.path.join(fusion_crf_save_root, name.replace('-A', '') + '.png')

                    img1_dir = img1_dir.replace('\\', '/')
                    img2_dir = img2_dir.replace('\\', '/')
                    decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
                    fusion_savePath = fusion_savePath.replace('\\', '/')

                    image1 = Image.open(img1_dir)
                    image2 = Image.open(img2_dir)

                    input_transform = transforms.Compose([
                        transforms.Grayscale(1),
                        transforms.ToTensor(),
                    ])

                    image1_tensor = input_transform(image1).unsqueeze(0)
                    image2_tensor = input_transform(image2).unsqueeze(0)

                    fft1 = util.FFT_Guass_test_v6(self.opt.R_Scale).tensorfft(image1_tensor).to(self.device)
                    fft2 = util.FFT_Guass_test_v6(self.opt.R_Scale).tensorfft(image2_tensor).to(self.device)

                    R1 = model_R(fft1)
                    R2 = model_R(fft2)

                    ifft1, high_frequency1, low_frequency1, complete_frequency1, low_filter1 = util.FFT_Guass_test_v6(
                        self.opt.R_Scale).tensorifft(image1_tensor, R1)
                    ifft2, high_frequency2, low_frequency2, complete_frequency2, low_filter2 = util.FFT_Guass_test_v6(
                        self.opt.R_Scale).tensorifft(image2_tensor, R2)

                    ifft = torch.cat((ifft1, ifft2), dim=1).to(self.device)

                    output_focus = model_T(ifft)

                    output = torch.softmax(output_focus, dim=1)

                    image_mask = torch.max(output.cpu(), 1)[1]


                    image_decisionMap = image_mask.numpy().squeeze(0) * 255
                    image_decisionMap[image_decisionMap < 0] = 0
                    image_decisionMap[image_decisionMap > 255] = 255

                    image_decisionMap = Image.fromarray(image_decisionMap.astype(np.uint8))
                    image_decisionMap.save(decisionMap_savePath)

                    image_mask = image_mask.numpy()
                    image_mask = np.transpose(image_mask, (1, 2, 0))

                    if len(np.array(image1).shape) == 3:
                        image_mask = np.repeat(image_mask, 3, 2)

                    if image_mask.shape != np.array(image1).shape:
                        img_shpae = np.array(image1).shape
                        image_mask = cv2.resize(image_mask.astype(float), (img_shpae[1], img_shpae[0]))

                    image_fusion = image1 * image_mask + image2 * (1 - image_mask)
                    image_fusion[image_fusion < 0] = 0
                    image_fusion[image_fusion > 255] = 255
                    image_fusion = Image.fromarray(image_fusion.astype(np.uint8))
                    image_fusion.save(fusion_savePath)

                    # CRF
                    fusion_morphology.fusion_main(decisionMap_savePath, img1_dir, img2_dir,
                                                  decisionMap_morphology_savePath,
                                                  decisionMap_crf_savePath, fusion_crf_savePath,
                                                  2, sdims1=9, compat1=9, sdims2=9,
                                                  schan=20, compat2=9)


            end_time = time.time()
            print("Average time consumption：", (end_time - begain_time) / 20)


if __name__ == '__main__':
    begain_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, default='./dataset/',
                        help='path of test dataset path in train')
    parser.add_argument('--dataset', default=['Lytro', 'MFFW'],
                        help='path of test dataset path in train')
    parser.add_argument('--ratio', type=int, default=0.1, help='filter ratio')
    parser.add_argument('--R_Scale', type=int, default=200, help='scale of R')
    parser.add_argument('--test_save_path', type=str, default='./results/test/train_model_v9_BS32_LRle-4_BCEWLoss_FFT_GuassNonormalR_200/',
                        help='path of save path in train')
    parser.add_argument('--test_file_ext', type=str, default='.tif',
                        help='[.jpg, .png, .tif]')
    parser.add_argument('--test_model_R', type=str, default='./models/train_model_v9_BS32_LRle-4_BCEWLoss_FFT_GuassNonormalR_200/R.pkl', help='path of save path in train')
    parser.add_argument('--test_model_T', type=str, default='./models/train_model_v9_BS32_LRle-4_BCEWLoss_FFT_GuassNonormalR_200/T.pkl', help='path of save path in train')
    parser.add_argument('--cuda', default=True, help='enables cuda')

    Main(parser.parse_args()).funm()
    end_time = time.time()
    print("Average time consumption：", (end_time - begain_time) / 20)