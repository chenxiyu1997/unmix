import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from evaluation import compute_rmse, compute_sad, close_position
from utils import print_args, SparseLoss, NonZeroClipper, MinVolumn
from data_loader import set_loader
from model import Init_Weights, MUNet
from datetime import datetime
import tempfile
from io import BytesIO
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import argparse
import random
import time
import os

class Config:
    def __init__(self):
        self.num_classes = 0
        self.band = 0
        self.col = 0
        self.row = 0
        self.fix_random = False  # 默认不修复随机性
        self.seed = 0  # 随机种子
        self.gpu_id = '0,1,2'  # GPU ID
        self.batch_size = 128  # 批量大小
        self.patch = 1  # 输入数据大小
        self.learning_rate_en = 0.0003  # 编码器学习率
        self.learning_rate_de = 0.0001  # 解码器学习率
        self.weight_decay = 1e-5  # 网络参数正则化
        self.lamda = 0.03  # 稀疏正则化
        self.reduction = 2  # 压缩减少
        self.delta = 1.0  # delta系数
        self.gamma = 0.8  # 学习率衰减
        self.epoch = 50  # 训练周期
        self.dataset = 'muffle'  # 默认使用的数据集

def getInfo(image_data):
    #row col num_classes band
    return image_data['label'].shape[0], image_data['label'].shape[1], image_data['label'].shape[2], image_data['Y'].shape[2]

def previewImages(abu_est, label, M_true, edm_result, num_classes):
    # abundance map
    for i in range(abu_est.shape[0]):
        plt.subplot(3, num_classes, i+1)
        plt.imshow(abu_est[i,:,:])
        plt.subplot(3, num_classes, i+1+num_classes)
        plt.imshow(label[i,:,:])
        plt.subplot(3, num_classes, i + 1 + num_classes * 2)
        plt.plot(edm_result[:, i])
    plt.show()

def modelEvaluation(net, test_loaders, num_classes, label, M_true, band, row, col):
    net.eval()
    print(net.spectral_se)
    for i, testdata in enumerate(test_loaders):
        x, y = testdata
        x = x.cuda()
        y = y.cuda()

        abu, output = net(x, y)

    # compute metric
    abu_est = torch.reshape(abu.squeeze(-1).permute(2,1,0), (num_classes,row,col)).permute(0,2,1).cpu().data.numpy()
    edm_result = torch.reshape(net.decoder[0].weight, (band,num_classes)).cpu().data.numpy()

    position = close_position(M_true, edm_result)
    abu_est = abu_est[position,:,:]
    edm_result = edm_result[:,position]

    #RMSE（Root Mean Square Error，均方根误差）和SAD（Spectral Angle Distance，光谱角距离)
    RMSE = compute_rmse(label, abu_est)
    SAD = compute_sad(M_true, edm_result)
    print('**********************************')
    print('RMSE: {:.5f} | SAD: {:.5f}'.format(RMSE, SAD))
    print('**********************************')

    return abu_est, edm_result, RMSE, SAD

def training(band, num_classes, ldr_dim, M_init, train_loaders, args):
    net = MUNet(band, num_classes, ldr_dim, args.reduction).cuda()
    
    # initialize net parameters and endmembers
    Init_Weights(net,'xavier', 1)

    net_dict = net.state_dict()
    net_dict['decoder.0.weight'] = M_init
    net.load_state_dict(net_dict)

    # loss funtion and regularization
    apply_nonegative = NonZeroClipper()
    loss_func = nn.MSELoss()
    criterionSparse = SparseLoss(args.lamda)
    criterionVolumn = MinVolumn(band, num_classes, args.delta)

    # optimizer setting
    params = map(id, net.decoder.parameters())
    ignored_params = list(set(params))      
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters()) 
    optimizer = torch.optim.Adam([{'params': base_params},{'params': net.decoder.parameters(), 'lr': args.learning_rate_de}],
                                    lr = args.learning_rate_en, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.gamma)

    for epoch in range(args.epoch):
        for i, traindata in enumerate(train_loaders):        
            net.train()

            x, y = traindata       
            x = x.cuda()
            y = y.cuda()
            
            abu, output = net(x,y)            
            output = torch.reshape(output, (output.shape[0], band))
            x = torch.reshape(x, (output.shape[0], band))

            # reconstruction loss
            MSE_loss = torch.mean(torch.acos(torch.sum(x * output, dim=1)/
                        (torch.norm(output, dim=1, p=2)*torch.norm(x, dim=1, p=2))))
            # sparsity and minimum volume regularization
            MSE_loss += criterionSparse(abu) + criterionVolumn(net.decoder[0].weight)       

            optimizer.zero_grad()
            MSE_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1) 
            optimizer.step()
            net.decoder.apply(apply_nonegative)    
        
        if epoch % 1 == 0:
            print('Epoch: {:d} | Train Unmix Loss: {:.5f} | RE Loss: {:.5f} | Sparsity Loss: {:.5f} | Minvol: {:.5f}'
                .format(epoch, MSE_loss, loss_func(output, x), criterionSparse(abu), criterionVolumn(net.decoder[0].weight)))

        scheduler.step()
    
    return net

def save_abu_images(save_dir, abu_est, edm_result, num_images):
    """
    将abu_est数组中的图像保存到指定的文件夹下。

    Parameters:
    - save_dir: 存储图像的目录路径。
    - abu_est: 包含要保存图像的numpy数组。
    - num_images: 要保存的图像数量。
    """
    # 确保交互模式关闭
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    matplotlib.use('Agg')
    plt.ioff()

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    abuImages = []
    edmImages = []

    # 遍历并保存每个图像
    for i in range(num_images):
        plt.imshow(abu_est[i, :, :])
        plt.axis('off')  # 移除坐标轴

        # 构建文件保存路径
        save_path = os.path.join(save_dir, f'abu_image_{i}.png').replace("\\", "/")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        abuImages.append(save_path)

        # 清除当前图形，避免重叠
        plt.clf()

    # 遍历并保存每个曲线
    for i in range(num_images):
        plt.plot(edm_result[:, i])

        # 构建文件保存路径
        save_path = os.path.join(save_dir, f'edm_image_{i}.png').replace("\\", "/")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        edmImages.append(save_path)

        # 清除当前图形，避免重叠
        plt.clf()

    # 重新启用交互模式，如果你后续需要
    plt.ion()

    return abuImages, edmImages

from fpdf import FPDF

def generate_pdf_with_images_and_errors(report_title, lidar, image, abundance_images_real, abundance_images_unmixed, spectrum_images_real, spectrum_images_unmixed, errors, output_filename="image/report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, report_title, 0, 1, 'C')

    # 图片宽度、高度和边距
    img_width = 60
    img_height = 45
    margin = 10
    space_between_images = 5

    def add_images_side_by_side(image_data_left, image_data_right):
        img_width = 80

        # 使用PIL计算图片的高度
        image_left = Image.open(BytesIO(image_data_left))
        width_left, height_left = image_left.size
        img_height = img_width * height_left / width_left
        
        margin = 10

        # 计算两张图片并排放置时的起始x坐标
        start_x_left = (pdf.w - 2 * img_width - margin) / 2
        start_x_right = start_x_left + img_width + margin

        # 创建临时文件并写入LIDAR图片数据
        fd_left, path_left = tempfile.mkstemp(suffix='.png')
        with os.fdopen(fd_left, 'wb') as tmp_left:
            tmp_left.write(image_data_left)

        # 创建临时文件并写入高光谱图片数据
        fd_right, path_right = tempfile.mkstemp(suffix='.png')
        with os.fdopen(fd_right, 'wb') as tmp_right:
            tmp_right.write(image_data_right)

        # 插入图片到PDF
        pdf.image(path_left, x=start_x_left, y=pdf.get_y() + 10, w=img_width, h=img_height)
        pdf.image(path_right, x=start_x_right, y=pdf.get_y() + 10, w=img_width, h=img_height)

        # 更新PDF的y坐标，为后续内容腾出空间
        pdf.set_y(pdf.get_y() + img_height + 20)

        # 清理临时文件
        os.remove(path_left)
        os.remove(path_right)

    # 首先添加LIDAR和高光谱图片
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 10, "Lidar and Hyperspectral", 0, 1)
    add_images_side_by_side(lidar, image)

    def add_images_section(title, images_pair_list):
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 10, title, 0, 1)
        for real_img, unmixed_img in images_pair_list:
            # 图片总宽度包括图片本身的宽度和它们之间的空间
            total_img_width = img_width * 2 + space_between_images
            # 计算起始 x 坐标以使图片组居中
            x_left = (pdf.w - total_img_width) / 2
            y_start = pdf.get_y() + 2  # 在文本和图片之间添加一点垂直空间
            
            # 检查是否需要添加新的页面
            if y_start + img_height + 10 > pdf.h - margin:
                pdf.add_page()
                y_start = margin + 10  # 重置 y_start 为顶部边距加上标题高度
                
            # 在图片上方添加描述
            pdf.set_xy(x_left, y_start)
            pdf.cell(img_width, 10, 'Real', 0, 0, 'C')
            pdf.set_x(x_left + img_width + space_between_images)
            pdf.cell(img_width, 10, 'Unmixed', 0, 1, 'C')
            y_start += 10  # 更新 y_start 以包含描述的高度
            
            # 添加真实和解混后的图片
            pdf.image(real_img, x=x_left, y=y_start, w=img_width, h=img_height)
            pdf.image(unmixed_img, x=x_left + img_width + space_between_images, y=y_start, w=img_width, h=img_height)
            
            # 更新 y 坐标以放置下一对图片
            new_y = y_start + img_height + margin
            pdf.set_y(new_y)

    # 添加丰度图和端元曲线图对比部分
    add_images_section(f"Abundance Maps Comparison (RMSE: {errors['RMSE']:.4f}):", zip(abundance_images_real, abundance_images_unmixed))
    add_images_section(f"Endmember Spectra Comparison (SAD: {errors['SAD']:.4f}):", zip(spectrum_images_real, spectrum_images_unmixed))

    # # 误差数据部分
    # pdf.add_page()
    # pdf.cell(0, 10, 'Error Data:', 0, 1)
    # for error_name, error_value in errors.items():
    #     pdf.cell(0, 10, f"{error_name}: {error_value:.5f}", 0, 1)

    # 保存 PDF
    pdf.output(output_filename)

def try_get_lidar_png(input_data):
    # 检查MPN_png是否存在
    if 'MPN_png' not in input_data:
        # 生成一张灰色的PNG图片
        # 创建一个灰色的100x100像素图像，灰色的RGB值为(128, 128, 128)
        gray_image = Image.new('RGB', (100, 100), (128, 128, 128))
        # 将图像保存到二进制流中，格式为PNG
        img_byte_arr = BytesIO()
        gray_image.save(img_byte_arr, format='PNG')
        # 获取二进制PNG数据
        mpn_png_data = img_byte_arr.getvalue()
    else:
        mpn_png_data = input_data['MPN_png']

    return mpn_png_data

def unmixing(config, input_data):
    # create dataset and model
    train_loaders, test_loaders, label, M_init, M_true, num_classes, band, col, row, ldr_dim = set_loader(config, input_data)

    # model training
    net = training(band, num_classes, ldr_dim, M_init, train_loaders, config)

    # model evaluation
    abu_est, edm_result, rmse, sad = modelEvaluation(net, test_loaders, num_classes, label, M_true, band, row, col)

    # preview result
    # previewImages(abu_est, label, M_true, edm_result, num_classes)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    abuImages, edmImages = save_abu_images("image/unmix/" + time_str, abu_est, edm_result, num_classes)
    abuRealImages, edmRealImages = save_abu_images("image/real/" + time_str, label, M_true, num_classes)

    errors = {'RMSE': rmse, 'SAD': sad}
    generate_pdf_with_images_and_errors("Hyperspectral Unmixing Data Report", try_get_lidar_png(input_data), input_data['Y_png'], abuRealImages, abuImages, edmRealImages, edmImages, errors)

    unmixRes = {'rmse': rmse, 'sad': sad, 'abuImages': abuImages, 'edmImages': edmImages}

    return unmixRes
