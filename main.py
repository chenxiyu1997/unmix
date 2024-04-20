from flask import jsonify
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from evaluation import compute_rmse, compute_sad
from utils import print_args, SparseLoss, NonZeroClipper, MinVolumn
from data_loader import set_loader
from model import Init_Weights, MUNet
from unmixing import Config, unmixing
from database import initDatabase, insert_lidar_data, delete_lidar_data, find_lidar_data, insert_image_data, delete_image_data, find_image_data, get_names_from_table, blur_image, blur_lidar, denoise_image, sharpen_image, add_user, add_user_lidar, add_user_image_data, get_user_lidars, get_user_image_data, delete_user, get_user_info, get_all_unmixing_records_by_name, get_user_image_datas, get_user_lidar_datas

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

def creatTestDataBase():
    users = [
        ("alice_wonder", "hash1", "Female", 28, "1234567890", "alice@example.com", "admin"),
        ("bob_builder", "hash2", "Male", 35, "1234567891", "bob@example.com", "user"),
        ("carol_singer", "hash3", "Female", 30, "1234567892", "carol@example.com", "user"),
        ("dave_runner", "hash4", "Male", 40, "1234567893", "dave@example.com", "user"),
        ("eve_hacker", "hash5", "Female", 32, "1234567894", "eve@example.com", "admin")
    ]

    for user in users:
        add_user(*user)

    # 生成测试数据
    image_file = r'./data/muffle_dataset_130_90.mat'
    input_data = sio.loadmat(image_file)
    image_file = r'./data/houston_170_dataset.mat'
    input_data1 = sio.loadmat(image_file)

    initDatabase()
    name = "muffle"
    name1 = "houston"

    # 存储雷达数据
    insert_lidar_data(name, input_data['MPN'])
    insert_lidar_data(name1, input_data1['MPN'])

    # 存储图像及相关数据
    insert_image_data(
        name,
        input_data['Y'],
        input_data['label'],
        input_data['M1'],
        input_data['M']
    )

    insert_image_data(
        name1,
        input_data1['Y'],
        input_data1['label'],
        input_data1['M1'],
        input_data1['M']
    )

    # 模糊 锐化 降噪 图像
    blur_lidar(name, "muffleBlur", 2)
    blur_image(name, "muffleBlur", 2)
    sharpen_image(name, "muffleSharpen", 2)
    denoise_image(name, "muffleDenoise", 2)

    blur_lidar(name1, "houstonBlur", 2)
    blur_image(name1, "houstonBlur", 2)
    sharpen_image(name1, "houstonSharpen", 2)
    denoise_image(name1, "houstonDenoise", 2)

    assignments = [
        ("alice_wonder", "muffle"),
        ("alice_wonder", "muffleBlur"),
        ("alice_wonder", "muffleDenoise"),
        ("bob_builder", "houston"),
        ("bob_builder", "houstonBlur"),
        ("bob_builder", "houstonSharpen"),
        ("bob_builder", "houstonDenoise"),
        ("carol_singer", "houstonBlur"),
        ("dave_runner", "muffleBlur"),
        ("eve_hacker", "houstonSharpen"),
        ("shareSpace", "muffle"),
        ("shareSpace", "muffleBlur"),
        ("shareSpace", "muffleDenoise"),
        ("shareSpace", "houston"),
        ("shareSpace", "houstonBlur"),
    ]

    # 执行分配
    for username, data_name in assignments:
        add_user_image_data(username, data_name)

    assignments = [
        ("alice_wonder", "muffle"),
        ("bob_builder", "houston"),
        ("carol_singer", "muffle"),
        ("dave_runner", "houston"),
        ("eve_hacker", "muffle")
    ]

    # 执行分配
    for username, data_name in assignments:
        add_user_lidar(username, data_name)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    torch.cuda.set_device(0)

    initDatabase()

    creatTestDataBase()

    image_file = r'./data/muffle_dataset_130_90.mat'
    input_data = sio.loadmat(image_file)
    image_file = r'./data/houston_170_dataset.mat'
    input_data1 = sio.loadmat(image_file)

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

    name = "muffle"
    # 从数据库中查找数据
    lidar_data = find_lidar_data(name)
    image_data = find_image_data("muffle")
    
    # 构建并返回input_data字典
    input_data_out = combined_dict = {**lidar_data, **image_data}
    input_data_out["username"] = "admin"

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