import torchvision.transforms as transforms
from torch.autograd import Variable

from dataset import datasetList_src
from torch.utils.data import DataLoader
import torch
import os
import datetime
from PIL import Image
import numpy as np
from itertools import cycle
import torch.nn as nn
import argparse
import pandas as pd
from utils import data_transforms_label, util
from models import model_v9 as model
import cv2

import sys


def main(opt):
    # ************************************************ 1 **********************************************
    image_size = opt.image_size
    input_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    gray_transform = transforms.Compose([
        transforms.Grayscale(1)
    ])

    co_transform_label = data_transforms_label.Compose([
        data_transforms_label.RandomVerticalFlip(),
        data_transforms_label.RandomHorizontalFlip(),
    ])


    # ************************************************ 2 **********************************************
    # ================================= load train data =============================
    train_set_label = datasetList_src.data_set(csv_path=opt.dataset_dir,
                                           input_transform=input_transform,
                                           target_transform=target_transform,
                                           gray_transform=gray_transform,
                                           co_transform=co_transform_label,
                                           ratio=opt.ratio)
    print(len(train_set_label))
    train_loader_label = DataLoader(train_set_label, batch_size=opt.batch_size_label, shuffle=True,
                                    num_workers=opt.workers)

    # ================================= train note =============================
    if not os.path.exists(opt.lossDir):
        os.makedirs(opt.lossDir)
    df = pd.DataFrame(columns=['time', 'step', 'loss', 'Accuracy'])
    df.to_csv(opt.lossDir + "loss.csv", index=False)
    # ====================================== train ==================================
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    print(device)
    model_R = model.RNet(in_ch=2)
    model_T = model.TaskNetwork(in_ch=2, out_ch=2)
    save_dir = opt.save_models

    # Find the latest model and continue training from the latest model. This avoids starting training from the first one after each pause.
    # pkl: suffix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    initial_epoch = util.findLastCheckpoint(save_dir=save_dir, modelType='T')  # load the last models in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model_T = torch.load(os.path.join(save_dir, 'T_%03d.pkl' % initial_epoch))
    print("model: ", model_T)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    initial_epoch = util.findLastCheckpoint(save_dir=save_dir, modelType='R')  # load the last models in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model_R = torch.load(os.path.join(save_dir, 'R_%03d.pkl' % initial_epoch))
    print("model: ", model_T)

    model_R.to(device)
    model_T.to(device)
    model_T.train()
    model_R.train()
    loss_function_T = torch.nn.BCEWithLogitsLoss()
    optimizer_T = torch.optim.Adam(model_T.parameters(), lr=opt.learning_rate)  # 0.92-0.99
    optimizer_R = torch.optim.Adam(model_R.parameters(), lr=opt.learning_rate)  # 0.92-0.99
    scheduler_T = torch.optim.lr_scheduler.StepLR(optimizer_T, step_size=10, gamma=0.5)
    scheduler_R = torch.optim.lr_scheduler.StepLR(optimizer_R, step_size=10, gamma=0.5)
    train_label(model_T, model_R, opt.Epoch, initial_epoch, train_loader_label, loss_function_T, optimizer_T, optimizer_R, scheduler_T, scheduler_R, device, opt)


def train_label(model_T, model_R, EPOCH, initial_epoch, train_loader_label, loss_function_T,  optimizer_T, optimizer_R, scheduler_T, scheduler_R, device, opt):
    for epoch in range(initial_epoch, EPOCH):
        # 1.Train
        train_loss_T = []
        train_accuracy = 0
        for i, data in enumerate(train_loader_label):
            source_img1, source_img2, source_img3, label_onehot, label = data

            # img_foreground, img_background, img_groundTruth, label_one_hot, label
            source_img1, source_img2, source_img3, label_onehot, label = source_img1.to(device), source_img2.to(device), source_img3.to(device), label_onehot.to(device), label.to(device)

            fft1 = util.FFT_Guass_train_v6(opt.R_Scale).tensorfft(source_img1 * 255).to(device)
            fft2 = util.FFT_Guass_train_v6(opt.R_Scale).tensorfft(source_img2 * 255).to(device)

            R1 = model_R(fft1).to(device)
            R2 = model_R(fft2).to(device)

            ifft_img1, high_frequency1, low_frequency1, complete_frequency1, low_filter1 = util.FFT_Guass_train_v6(opt.R_Scale).tensorifft(source_img1, R1)
            ifft_img2, high_frequency2, low_frequency2, complete_frequency2, low_filter2 = util.FFT_Guass_train_v6(opt.R_Scale).tensorifft(source_img2, R2)

            ifft_img = torch.cat((ifft_img1, ifft_img2), dim=1).to(device)

            output_Task = model_T(ifft_img)

            loss = loss_function_T(output_Task, label_onehot)

            optimizer_R.zero_grad()
            optimizer_T.zero_grad()
            loss.backward()
            optimizer_R.step()
            optimizer_T.step()
            train_loss_T.append(loss.item())
            # --------------
            #  Log Progress
            # --------------

            print()
            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d]  [T loss: %f]"
                % (epoch, opt.Epoch, i, len(train_loader_label),  loss.item())
            )

            # 7. train accuracy
            output = torch.softmax(output_Task, dim=1)
            pred_out = torch.max(output[:opt.batch_size_label, :, :, :], 1)[1]
            train_accuracy += torch.eq(pred_out, torch.squeeze(label).long()).sum().float().item()
            if i % 10 == 0:
                # print("Epoch:", epoch, " i:", i + 1, " train loss:", np.mean(train_loss))
                image_mask = torch.max(output[0].unsqueeze(0).cpu(), 1)[1]

                image_decisionMap = image_mask.numpy().squeeze(0) * 255

                image_decisionMap = Image.fromarray(image_decisionMap.astype(np.uint8))
                if not os.path.exists("./results/train_fusion"):
                    os.makedirs("./results/train_fusion")
                image_decisionMap.save("./results/train_fusion/decision.png")

        accuracy = train_accuracy / (len(train_loader_label.dataset) * (opt.image_size ** 2))

        print()
        sys.stdout.write(
            "[Epoch %d/%d]  [T average loss: %f]  [Train accuracy: %f]"
            % (epoch, opt.Epoch, np.mean(train_loss_T),  accuracy)
        )

        # 写入loss
        time = '%s' % datetime.datetime.now()
        step = 'step[%d]' % epoch
        Loss = '%f' % np.mean(train_loss_T)
        Accuracy = '%f' %accuracy
        losslist = [time, step, Loss, Accuracy]
        data = pd.DataFrame([losslist])
        if not os.path.exists(opt.lossDir):
            os.makedirs(opt.lossDir)
        data.to_csv(os.path.join(opt.lossDir, "loss.csv"), mode='a', header=False, index=False)

        scheduler_T.step()
        scheduler_R.step()

        # 4.save models
        model_savePath = opt.save_models
        if not os.path.exists(model_savePath):
            os.makedirs(model_savePath)

        torch.save(model_T, os.path.join(model_savePath, 'T_%03d.pkl' % (epoch + 1)))
        torch.save(model_R, os.path.join(model_savePath, 'R_%03d.pkl' % (epoch + 1)))
        # 5.train fusion

        img_root = opt.train_test_dataset_path
        save_root = opt.train_test_save_path

        img1_root = os.path.join(img_root, 'img1')
        img2_root = os.path.join(img_root, 'img2')
        img1_root = img1_root.replace('\\', '/')
        img2_root = img2_root.replace('\\', '/')

        decisionMap_save_root = os.path.join(save_root, 'decisionmap')
        fusion_save_root = os.path.join(save_root, 'fusion')

        decisionMap_save_root = decisionMap_save_root.replace('\\', '/')
        fusion_save_root = fusion_save_root.replace('\\', '/')
      

        if not os.path.exists(decisionMap_save_root):
            os.makedirs(decisionMap_save_root)
        if not os.path.exists(fusion_save_root):
            os.makedirs(fusion_save_root)
       
        img_names = os.listdir(img1_root)
        for img_name in img_names:
            name, suffix = img_name.split('.')
            img1_dir = os.path.join(img1_root, img_name)
            img2_dir = os.path.join(img2_root, img_name.replace('A', 'B'))

            decisionMap_savePath = os.path.join(decisionMap_save_root, str(epoch) + "_" + name.replace('-A', '') + '.png')
            fusion_savePath = os.path.join(fusion_save_root, str(epoch) + "_" + name.replace('-A', '') + '.png')

            img1_dir = img1_dir.replace('\\', '/')
            img2_dir = img2_dir.replace('\\', '/')
            decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
            fusion_savePath = fusion_savePath.replace('\\', '/')

            train_fusion(model_R, model_T, img1_dir, img2_dir, decisionMap_savePath, fusion_savePath, device, opt)


def train_fusion(model_R, train_model, image1_path, image2_path, decisionMap_savePath, fusion_savePath, device, opt):
    model_R.eval()
    train_model.eval()
    model_R.cpu()
    train_model.cpu()
    with torch.no_grad():
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        input_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

        image1_tensor = input_transform(image1).unsqueeze(0)
        image2_tensor = input_transform(image2).unsqueeze(0)

        fft1 = util.FFT_Guass_train_v6(opt.R_Scale).tensorfft(image1_tensor)
        fft2 = util.FFT_Guass_train_v6(opt.R_Scale).tensorfft(image2_tensor)

        R1 = model_R(fft1)
        R2 = model_R(fft2)

        ifft1, high_frequency1, low_frequency1, complete_frequency1, low_filter1 = util.FFT_Guass_train_v6(opt.R_Scale).tensorifft(image1_tensor, R1)
        ifft2, high_frequency2, low_frequency2, complete_frequency2, low_filter2 = util.FFT_Guass_train_v6(opt.R_Scale).tensorifft(image2_tensor, R2)

        ifft = torch.cat((ifft1, ifft2), dim=1)

        output_focus = train_model(ifft)

        output = torch.softmax(output_focus, dim=1)

        image_mask = torch.max(output.cpu(), 1)[1]

        image_decisionMap = image_mask.numpy().squeeze(0) * 255
        image_decisionMap[image_decisionMap < 0] = 0
        image_decisionMap[image_decisionMap > 255] = 255

        image_decisionMap = Image.fromarray(image_decisionMap.astype(np.uint8))
        image_decisionMap.save(decisionMap_savePath)

    model_R.to(device)
    train_model.to(device)
    model_R.train()
    train_model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Epoch', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='train learning rate')
    parser.add_argument('--batch_size_label', type=int, default=32, help='train batch size')
    parser.add_argument('--image_size', type=int, default=128, help='image size')
    parser.add_argument('--ratio', type=int, default=0.1, help='filter ratio')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--cuda_id', type=str, default="cuda:0", help='cuda id')

    parser.add_argument('--R_Scale', type=int, default="200", help='scale of R')

    parser.add_argument('--dataset_dir', type=str, default='./dataset/dataset_csv/data_all_v1.csv',
                        help='path of train dataset')
    parser.add_argument('--train_test_dataset_path', type=str, default='./dataset/Lytro/',
                        help='path of test dataset path in train')
    parser.add_argument('--train_test_save_path', type=str, default='./results/train/train_v1_model_v9_BS32_LRle-4_BCEWLoss_FFT_GuassNonormalR_200/',
                        help='path of save path in train')
    parser.add_argument('--lossDir', type=str, default='./results/train/train_v1_model_v9_BS32_LRle-4_BCEWLoss_FFT_GuassNonormalR_200/log/',
                        help='path of loss in train')
    parser.add_argument('--save_models', type=str, default='./models/train_v1_model_v9_BS32_LRle-4_BCEWLoss_FFT_GuassNonormalR_200/',
                        help='path of save path in train')
    parser.add_argument('--no_cuda', action='store_true', help="if set disables CUDA")

    main(parser.parse_args())