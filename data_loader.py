import torch
import torch.utils.data as Data
import numpy as np
import scipy.io as sio

def set_loader(args, input_data):
    image = input_data['Y']
    image = image.astype(np.float32)
    label = input_data['label']
    label = label.astype(np.float32).transpose(2,1,0)

    # 处理lidar/MPN
    if 'MPN' in input_data:
        lidar = input_data['MPN']
        lidar = lidar.astype(np.float32)
    else:
        # 如果MPN不存在，创建一个新的三维数组，尺寸与image的前两维一致，第三维长度为5，所有值设为0
        lidar = np.zeros((image.shape[0], image.shape[1], 5), dtype=np.float32)

    M_init = input_data['M1']
    M_init = torch.from_numpy(M_init).unsqueeze(2).unsqueeze(3).float() 
    M_true = input_data['M']

    train_point = []
    x_train = np.zeros((args.row*args.col, args.patch, args.patch, args.band), dtype=float)
    y_train = np.zeros((args.row*args.col, args.patch, args.patch, lidar.shape[2]), dtype=float)
    for i in range(args.row):
        for j in range(args.col):
            train_point.append([i,j])
    for k in range(len(train_point)):
        x_train[k,:,:,:] = gain_neighborhood_pixel(image, train_point, k, args.patch)
        y_train[k,:,:,:] = gain_neighborhood_pixel(lidar, train_point, k, args.patch)
        
    x_train = torch.from_numpy(x_train.transpose(0,3,1,2)).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train.transpose(0,3,1,2)).type(torch.FloatTensor)
    Label_train = Data.TensorDataset(x_train, y_train)
    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)

    label_test_loader = Data.DataLoader(Label_train, batch_size=args.row*args.col, shuffle=False)
    return label_train_loader, label_test_loader, label, M_init, M_true, args.num_classes, args.band, args.col, args.row, lidar.shape[2]


def mirror_hsi(height, width, band, edm, input_normalize, label_normalize, patch):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band), dtype=float)
    mirror_label=np.zeros((height+2*padding,width+2*padding,edm), dtype=float)
    #central region
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    mirror_label[padding:(padding+height),padding:(padding+width),:]=label_normalize
    #left region
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
        mirror_label[padding:(height+padding),i,:]=label_normalize[:,padding-i-1,:]
    #right region
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
        mirror_label[padding:(height+padding),width+padding+i,:]=label_normalize[:,width-1-i,:]
    #top region
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
        mirror_label[i,:,:]=mirror_label[padding*2-i-1,:,:]
    #bottom region
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]
        mirror_label[height+padding+i,:,:]=mirror_label[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("mirror_label shape : [{0},{1},{2}]".format(mirror_label.shape[0],mirror_label.shape[1],mirror_label.shape[2]))
    print("**************************************************")
    return mirror_hsi, mirror_label

def gain_neighborhood_pixel(mirror_image, train_point, i, patch):
    x = train_point[i][0]
    y = train_point[i][1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image
