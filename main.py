import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from evaluation import compute_rmse, compute_sad
from utils import print_args, SparseLoss, NonZeroClipper, MinVolumn
from data_loader import set_loader
from model import Init_Weights, MUNet
from unmixing import Config, unmixing
from database import initDatabase, insert_lidar_data, delete_lidar_data, find_lidar_data, insert_image_data, delete_image_data, find_image_data, blur_image, blur_lidar, denoise_image, sharpen_image

from datetime import datetime
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import argparse
import random
import time
import json
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    torch.cuda.set_device(0)

    image_file = r'./data/muffle_dataset_130_90.mat'
    # image_file = r'./data/houston_170_dataset.mat'
    input_data = sio.loadmat(image_file)

    initDatabase()
    name = "testName"
    blurName = "testBlur"
    
    # 存储雷达数据
    insert_lidar_data(name, input_data['MPN'])
    
    # 存储图像及相关数据
    insert_image_data(
        name,
        input_data['Y'],
        input_data['label'],
        input_data['M1'],
        input_data['M']
    )

    blur_lidar(name, blurName, 2)
    sharpen_image(name, blurName, 2)

    # 从数据库中查找数据
    lidar_data = find_lidar_data(name)
    image_data = find_image_data(name)
    
    # 构建并返回input_data字典
    input_data_out = combined_dict = {**lidar_data, **image_data}

    # 初始化 config
    config = Config()
    config.num_classes = input_data_out["num"]
    config.band = input_data_out["band"]
    config.col = input_data_out["col"]
    config.row = input_data_out["row"]
    config.learning_rate_en = 0.0003  # 编码器学习率
    config.learning_rate_de = 0.0001  # 解码器学习率
    config.lamda = 0.03  # 稀疏正则化
    config.reduction = 2  # 压缩减少
    config.delta = 1.0  # delta系数
    config.gamma = 0.8  # 学习率衰减
    config.epoch = 50  # 训练周期

    result = json.dumps(unmixing(config, input_data_out), indent=4)

    print(result)