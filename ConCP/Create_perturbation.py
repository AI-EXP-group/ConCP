import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Subset
import torchvision.models as models
import torch.nn.functional as F
from models import *
from models import googlenet_copy
from models import resnet
from models import mobilenet
import os
import csv
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util import *
from models import RES
import pandas as pd
random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

device = 'cuda:1'

'''
The path for target dataset and public out-of-distribution (POOD) dataset. The setting used 
here is CIFAR-10 as the target dataset and Tiny-ImageNet as the POOD dataset. Their directory
structure is as follows:

dataset_path--cifar-10-batches-py
            |
            |-tiny-imagenet-200
'''
dataset_path = '/home/ubuntu/Public/deeplearning/workspace/ymq/narcissus/data/'


def narcissus_gen(dataset_path = dataset_path, lab=0):
    #Noise size, default is full image size
    noise_size = 32

    #Radius of the L-inf ball
    l_inf_r = 64/255

    #Model for generating surrogate model and trigger
    # surrogate_model = torch.load("./checkpoint/resnet200_tinyimagenet.pth")
    # generating_model = googlenet_copy.GoogLeNet.to(device)

    #Surrogate model training epochs
    surrogate_epochs = 200

    #Learning rate for poison-warm-up
    generating_lr_warmup = 0.15
    warmup_round = 5

    #Learning rate for trigger generating
    generating_lr_tri = 0.01
    gen_round = 1000

    #Training batch size
    train_batch_size = 128

    #The model for adding the noise
    patch_mode = 'add'

    #The argumention use for all training set
    transform_before = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),  # cifar10
        # transforms.Normalize([0.4376,0.4437,0.4728], [0.1980,0.2010,0.1970]), #svhn
        # transforms.Normalize([0.3403, 0.3121, 0.3214], [0.1595, 0.1590, 0.1683])#gtsrb
    ])
    transform_after = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), #cifar10
        # transforms.Normalize([0.4376,0.4437,0.4728], [0.1980,0.2010,0.1970]), #svhn
        # transforms.Normalize([0.3403, 0.3121, 0.3214], [0.1595, 0.1590, 0.1683])#gtsrb
    ])
    ori_train = torchvision.datasets.CIFAR10(root=dataset_path, download=False,train=False,transform=transform_before)

    #Outter train dataset
    train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
    #Inner train dataset
    train_target_list = list(np.where(np.array(train_label)==lab)[0])
    print(len(train_target_list))
    num = int(len(train_target_list) * 0.5)
    train_target_list = train_target_list[:num]
    print(lab,len(train_target_list))
    train_sub_data = Subset(ori_train, train_target_list)
    train_target = CustomDataset(train_sub_data)

    # poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)
    trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)
    #num_workers是多线程读数据


    # Batch_grad
    condition = True
    noise = torch.zeros((1, 3, noise_size, noise_size), device=device)
    criterion = torch.nn.CrossEntropyLoss()

    #Prepare models and optimizers for poi_warm_up training
    print("开始了")
    poi_warm_up_model = torch.load("./checkpoint/googlenet200_tinyimagenet.pth")
    poi_warm_up_model=poi_warm_up_model.to(device)
    poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)
    #Poi_warm_up stage
    poi_warm_up_model.train()
    for param in poi_warm_up_model.parameters():
        param.requires_grad = True

    # Training the surrogate model
    print('Training the surrogate model')
    for epoch in range(0, 5):
        poi_warm_up_model.train()
        loss_list = []
        for images, labels in trigger_gen_loaders:
            images, labels = images.to(device), labels.to(device)
            outputs = poi_warm_up_model(images)
            loss = criterion(outputs, labels)
            poi_warm_up_model.zero_grad()
            poi_warm_up_opt.zero_grad()
            loss.backward()
            loss_list.append(float(loss.data))
            poi_warm_up_opt.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %e' % (epoch, ave_loss))


    #Trigger generating stage
    for param in poi_warm_up_model.parameters():
        param.requires_grad = False    #冻结代理模型参数
    batch_pert = torch.autograd.Variable(noise.to(device), requires_grad=True)
    batch_opt = torch.optim.RAdam(params=[batch_pert],lr=0.01)
    sche = torch.optim.lr_scheduler.StepLR(batch_opt, 100, 0.7)

    print('Training the trigger')
    # for minmin in tqdm.notebook.tqdm(range(gen_round)):    #数据集：目标类
    for minmin in range(1000):
        loss_list = []
        for images, labels in trigger_gen_loaders:
            images, labels = images.to(device), labels.to(device)
            new_label=torch.full_like(labels, lab)
            new_label=new_label.to(device)
            new_images = torch.clone(images)
            clamp_batch_pert = torch.clamp(batch_pert,-l_inf_r,l_inf_r)  #限制noise中值的范围
            new_images = torch.clamp(apply_noise_patch(clamp_batch_pert,new_images.clone(),mode=patch_mode),-1,1)
            #apply_noise_patch 将noise贴到图片上
            per_logits = poi_warm_up_model(new_images)
            loss = criterion(per_logits, new_label)
            loss_regu = torch.mean(loss)
            batch_opt.zero_grad()
            loss_list.append(float(loss_regu.data))
            loss_regu.backward()
            #retain_graph=True是保存计算图
            batch_opt.step()
        sche.step()
        ave_loss = np.average(np.array(loss_list))
        ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
        print(minmin,'Gradient:',ave_grad,'Loss:', ave_loss)
        if ave_grad == 0:
            break

    noise = torch.clamp(batch_pert,-l_inf_r,l_inf_r)
    best_noise = noise.clone().detach().cpu()
    # plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
    # plt.show()
    print('Noise max val:',noise.max())

    save_noise_path = './new_data/defense_trigger_64255/cifar_'+str(lab)+'.pth'
    torch.save(best_noise, save_noise_path)

    return best_noise