import base64
import hashlib
import json
import pickle
import sqlite3
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from skimage.restoration import denoise_bilateral
from skimage.filters import unsharp_mask


def initDatabase():
    # 连接到SQLite数据库
    # 如果数据库不存在，会自动在当前目录创建一个名为example.db的数据库文件
    conn = sqlite3.connect('database/hsi_data.db')

    # 创建一个Cursor对象，用于执行SQL语句
    c = conn.cursor()

    # 创建一个名为lidar的表，如果该表不存在
    c.execute('''CREATE TABLE IF NOT EXISTS lidar
                (name TEXT PRIMARY KEY, MPN BLOB, MPN_png BLOB,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # 创建一个名为image_data的表，如果该表不存在
    # 其中name作为唯一标识符
    c.execute('''CREATE TABLE IF NOT EXISTS image_data
                (name TEXT PRIMARY KEY, 
                Y BLOB,
                Y_png BLOB,
                label BLOB, 
                M1 BLOB, 
                M BLOB, 
                row INTEGER, 
                col INTEGER, 
                num INTEGER, 
                band INTEGER,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # 创建一个名为users的表
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        sex TEXT,
        age INTEGER,
        phone TEXT,
        email TEXT,
        permission TEXT
    )''')

    # 创建用户与lidar数据的关系表
    c.execute('''
    CREATE TABLE IF NOT EXISTS user_lidar (
        username TEXT,
        lidar_name TEXT,
        PRIMARY KEY (username, lidar_name),
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE,
        FOREIGN KEY (lidar_name) REFERENCES lidar(name) ON DELETE CASCADE
    )''')

    # 创建用户与高光谱数据的关系表
    c.execute('''
    CREATE TABLE IF NOT EXISTS user_image_data (
        username TEXT,
        image_data_name TEXT,
        PRIMARY KEY (username, image_data_name),
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE,
        FOREIGN KEY (image_data_name) REFERENCES image_data(name) ON DELETE CASCADE
    )''')

    # 创建解混记录表，将 abu_est 和 edm_result 设置为 BLOB 类型
    c.execute('''
        CREATE TABLE IF NOT EXISTS unmixing_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            high_spectral_data TEXT,
            lidar_data TEXT,
            username TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            abu_est BLOB,
            edm_result BLOB,
            deconv_config TEXT,
            rmse REAL,
            sad REAL
        )
    ''')

    # 提交事务
    conn.commit()
    conn.close()


# 用户相关信息存储
def add_user(username, password, sex, age, phone, email, permission):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        REPLACE INTO users (username, password_hash, sex, age, phone, email, permission)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, password_hash, sex, age, phone, email, permission))
    conn.commit()

def add_user_lidar(username, lidar_name):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        REPLACE INTO user_lidar (username, lidar_name)
        VALUES (?, ?)
    ''', (username, lidar_name))
    conn.commit()

def add_user_image_data(username, image_data_name):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        REPLACE INTO user_image_data (username, image_data_name)
        VALUES (?, ?)
    ''', (username, image_data_name))
    conn.commit()

def get_user_lidars(username):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT lidar_name FROM user_lidar WHERE username = ?
    ''', (username,))
    return cursor.fetchall()

def get_user_images(username):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT image_data_name FROM user_image_data WHERE username = ?
    ''', (username,))
    return cursor.fetchall()

def get_user_image_datas(username):
    names = get_user_images(username)
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    all_data = []
    for name in names:
        data = find_image_data(name[0])  # 获取每张图片的详细信息
        if data:
            # 只有当有详细信息时才添加
            del data['Y']
            del data['label']
            del data['M1']
            del data['M']
            data['Y_png'] = base64.b64encode(data['Y_png']).decode('utf-8')
            all_data.append({'name': name, 'data': data})
    
    return all_data

def get_user_lidar_datas(username):
    names = get_user_lidars(username)
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    all_data = []
    for name in names:
        data = find_lidar_data(name[0])  # 获取每张图片的详细信息
        if data:
            # 只有当有详细信息时才添加
            data['row'] = data['MPN'].shape[0]
            data['col'] = data['MPN'].shape[1]
            del data['MPN']
            data['MPN_png'] = base64.b64encode(data['MPN_png']).decode('utf-8')
            all_data.append({'name': name[0], 'data': data})
    
    return all_data


def get_user_image_data(username):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT image_data_name FROM user_image_data WHERE username = ?
    ''', (username,))
    return cursor.fetchall()

def delete_user(username):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM users WHERE username = ?
    ''', (username,))
    conn.commit()

def delete_user_lidar(username, lidar_name):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM user_lidar WHERE username = ? AND lidar_name = ?
    ''', (username, lidar_name))
    conn.commit()

def delete_user_image_data(username, image_data_name):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM user_image_data WHERE username = ? AND image_data_name = ?
    ''', (username, image_data_name))
    conn.commit()

def get_user_info(username):
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT username, sex, age, phone, email, permission FROM users WHERE username = ?
    ''', (username,))
    return cursor.fetchone()

# 高光谱及雷达数据存储
def get_names_from_table(table_name):
    conn = sqlite3.connect('database/hsi_data.db')  # 替换为你的数据库路径
    c = conn.cursor()

    # 根据表名查询所有name
    c.execute(f"SELECT name FROM {table_name}")
    names = c.fetchall()  # 获取查询结果

    conn.close()  # 关闭数据库连接
    return [name[0] for name in names]

def insert_lidar_data(name, mpn_data):
    # 将NumPy数组序列化为二进制格式
    mpn_blob = pickle.dumps(mpn_data)
    
    # 连接到数据库
    conn = sqlite3.connect('database/hsi_data.db')
    cursor = conn.cursor()
    
    # 插入数据
    cursor.execute('''
    REPLACE INTO lidar (name, MPN, MPN_png)
    VALUES (?, ?, ?)''', (name, mpn_blob, lidar_to_png(mpn_data)))
    
    # 提交并关闭连接
    conn.commit()
    conn.close()

def delete_lidar_data(name):
    # 连接数据库
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    
    # 执行删除
    c.execute("DELETE FROM lidar WHERE name = ?", (name,))
    
    # 提交并关闭
    conn.commit()
    conn.close()

def find_lidar_data(name):
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    
    # 执行查询
    c.execute("SELECT * FROM lidar WHERE name = ?", (name,))
    result = c.fetchone()
    
    conn.close()
    
    if result:
        # 反序列化BLOB数据
        lidar_data = {
            'lidar_name': name,
            'MPN': pickle.loads(result[1]),
            'MPN_png': result[2],
            'upload_time': result[3],
        }
        return lidar_data
    else:
        return None

def insert_image_data(name, Y, label, M1, M):
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    
    # 序列化BLOB数据
    Y_blob = pickle.dumps(Y, protocol=pickle.HIGHEST_PROTOCOL)
    Y_png = image_to_png(Y)
    label_blob = pickle.dumps(label, protocol=pickle.HIGHEST_PROTOCOL)
    M1_blob = pickle.dumps(M1, protocol=pickle.HIGHEST_PROTOCOL)
    M_blob = pickle.dumps(M, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 插入数据
    c.execute("REPLACE INTO image_data (name, Y, Y_png, label, M1, M, row, col, num, band) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
              (name, Y_blob, Y_png, label_blob, M1_blob, M_blob, label.shape[0], label.shape[1], label.shape[2], Y.shape[2]))
    
    conn.commit()
    conn.close()


def delete_image_data(name):
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    
    c.execute("DELETE FROM image_data WHERE name = ?", (name,))
    
    conn.commit()
    conn.close()

def find_image_data(name):
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    
    c.execute("SELECT * FROM image_data WHERE name = ?", (name,))
    result = c.fetchone()
    
    conn.close()
    
    if result:
        # 反序列化BLOB数据
        data = {
            'image_name': name,
            'Y': pickle.loads(result[1]),
            'Y_png' : result[2],
            'label': pickle.loads(result[3]),
            'M1': pickle.loads(result[4]),
            'M': pickle.loads(result[5]),
            'row': result[6],
            'col': result[7],
            'num' : result[8],
            'band': result[9],
            'upload_time': result[10],
        }
        return data
    else:
        return None

def image_to_png(hyperspectral_data):
    # 假设 `hyperspectral_data` 是你的高光谱数据，形状为 [高度, 宽度, 波段数]
    # 例如，hyperspectral_data.shape 可能是 (100, 100, 20)

    # 重塑数据为 [像素数, 波段数]
    num_samples = hyperspectral_data.shape[0] * hyperspectral_data.shape[1]
    num_bands = hyperspectral_data.shape[2]
    data_reshaped = hyperspectral_data.reshape((num_samples, num_bands))

    # 应用PCA
    pca = PCA(n_components=3) # 提取三个主成分
    principal_components = pca.fit_transform(data_reshaped)

    # 归一化主成分到0-1范围
    scaler = MinMaxScaler()
    principal_components_scaled = scaler.fit_transform(principal_components)

    # 重塑为原始图像形状
    rgb_image = principal_components_scaled.reshape(hyperspectral_data.shape[0], hyperspectral_data.shape[1], 3)
    rgb_image = np.clip(rgb_image, 0, 1)

    # 将RGB图像保存为PNG格式的BLOB
    buf = BytesIO()
    plt.imsave(buf, rgb_image, format='png')
    png_blob = buf.getvalue()
    buf.close()
    
    return png_blob

def lidar_to_png(lidar):
    # 获取原始雷达图片
    original_radar_image = lidar[:, :, lidar.shape[2] - 1]

    # 将RGB图像保存为PNG格式的BLOB
    buf = BytesIO()
    plt.imsave(buf, original_radar_image, format='png')
    png_blob = buf.getvalue()
    buf.close()

    return png_blob

#操作记录
def add_deconvolution_record(high_spectral_data, lidar_data, username, abu_est, edm_result, deconv_config, rmse, sad):
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    # 序列化 abu_est 和 edm_result
    abu_est_blob = pickle.dumps(abu_est)
    edm_result_blob = pickle.dumps(edm_result)
    deconv_config_json = json.dumps(deconv_config)
    
    c.execute('''
        INSERT INTO unmixing_records (
            high_spectral_data, lidar_data, username, abu_est, edm_result, deconv_config, rmse, sad
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (high_spectral_data, lidar_data, username, abu_est_blob, edm_result_blob, deconv_config_json, rmse, sad))
    
    conn.commit()
    conn.close()

def delete_deconvolution_record(record_id):
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    c.execute('''
        DELETE FROM unmixing_records WHERE id = ?
    ''', (record_id,))
    conn.commit()
    conn.close()

def get_all_unmixing_records_by_name(username):
    conn = sqlite3.connect('database/hsi_data.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, high_spectral_data, lidar_data, username, timestamp, rmse, sad 
        FROM unmixing_records WHERE username = ?
    ''', (username,))
    records = c.fetchall()
    conn.close()

    # Convert records to dictionaries
    records_dict = []
    for record in records:
        record_dict = {
            'id': record[0],
            'high_spectral_data': record[1],
            'lidar_data': record[2],
            'username': record[3],
            'timestamp': record[4],
            'rmse': record[5],
            'sad': record[6]
        }
        records_dict.append(record_dict)

    return records_dict

# 数据处理
def gaussian_blur_multiband(image_data, sigma=1):
    """
    对三维NumPy数组应用高斯模糊，其中最后一个维度是波段。
    
    参数:
    - image_data: 三维NumPy数组，形状为(height, width, bands)
    - sigma: 高斯核的标准差，控制模糊程度
    
    返回:
    - 模糊后的图像数组，形状和输入相同
    """
    # 初始化一个和输入数组形状相同的数组来存储结果
    blurred_image = np.zeros_like(image_data)
    
    # 遍历每个波段，分别应用高斯模糊
    for band in range(image_data.shape[2]):
        blurred_image[:, :, band] = gaussian_filter(image_data[:, :, band], sigma=sigma)
    
    return blurred_image

def calculate_gradient_magnitude(image, sigma=1):
    """
    计算高光谱图像的综合梯度幅值。
    
    :param image: 高光谱图像，形状为 (高度, 宽度, 光谱波段数) 的三维数组。
    :param sigma: 高斯梯度幅值计算中使用的sigma值。
    :return: 二维梯度幅值图。
    """
    # 初始化一个空数组来存储每个波段的梯度幅值
    gradient_magnitudes = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    
    # 对每个波段独立计算梯度幅值
    for i in range(image.shape[2]):
        gradient_magnitudes[:, :, i] = gaussian_gradient_magnitude(image[:, :, i], sigma=sigma)
    
    # 沿光谱维度取平均或最大值来估算综合的梯度幅值
    # 你可以根据需要选择使用平均值或最大值
    # composite_gradient_magnitude = np.mean(gradient_magnitudes, axis=2)
    composite_gradient_magnitude = np.max(gradient_magnitudes, axis=2)
    
    return composite_gradient_magnitude

def bilateral_filter_multiband(image, sigma_color, sigma_spatial):
    """
    对高光谱图像的每个波段独立应用双边滤波进行降噪。

    :param image: 高光谱图像，形状为 (height, width, bands)。
    :param sigma_color: 色彩空间的标准差，用于双边滤波。
    :param sigma_spatial: 空间标准差，用于双边滤波。
    
    :return: 经过双边滤波降噪的高光谱图像。
    """
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
        # 对每个波段应用双边滤波
        filtered_image[:, :, i] = denoise_bilateral(image[:, :, i], sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    return filtered_image

def unsharp_mask_multiband(image, radius, amount):
    """
    对高光谱图像的每个波段独立应用unsharp_mask进行锐化。

    :param image: 高光谱图像，形状为 (height, width, bands)。
    :param radius: unsharp_mask的半径。
    :param amount: unsharp_mask的强度。
    
    :return: 经过unsharp_mask锐化的高光谱图像。
    """
    sharpened_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
        # 对每个波段应用unsharp_mask进行锐化
        sharpened_image[:, :, i] = unsharp_mask(image[:, :, i], radius=radius, amount=amount)

    filtered_image = np.clip(sharpened_image, 0.001, 1)
    return sharpened_image

def blur_lidar(original_name, new_name, sigma=1):
    # 读取指定名字的数据
    original_data = find_lidar_data(original_name)
    
    if not original_data:
        print("Original data not found.")
        return


    # 应用高斯模糊
    MPN = gaussian_blur_multiband(original_data['MPN'], sigma=sigma)
    
    # 使用高斯模糊后的MPN数据，以及原始的其他数据，保存到数据库中新的记录
    insert_lidar_data(new_name, MPN)
    
    print(f"Processed data saved with name: {new_name}")

def blur_image(original_name, new_name, sigma=1):
    # 读取指定名字的数据
    original_data = find_image_data(original_name)
    
    if not original_data:
        print("Original data not found.")
        return

    # 应用高斯模糊
    Y_blurred = gaussian_blur_multiband(original_data['Y'], sigma=sigma)
    label_blurred = gaussian_blur_multiband(original_data['label'], sigma=sigma)
    
    # 使用高斯模糊后的Y和label数据，以及原始的其他数据，保存到数据库中新的记录
    insert_image_data(new_name, Y_blurred, label_blurred, original_data['M1'], original_data['M'])
    
    print(f"Processed data saved with name: {new_name}")

def denoise_image(original_name, new_name, sigma_color = 1, sigma_spatial = 1):
    # 读取指定名字的数据
    original_data = find_image_data(original_name)
    
    if not original_data:
        print("Original data not found.")
        return

    # 应用各向异性降噪
    Y_blurred = bilateral_filter_multiband(original_data['Y'], sigma_color,sigma_spatial)
    label_blurred = bilateral_filter_multiband(original_data['label'], sigma_color,sigma_color)
    
    # 使用各向异性降噪后的Y和label数据，以及原始的其他数据，保存到数据库中新的记录
    insert_image_data(new_name, Y_blurred, label_blurred, original_data['M1'], original_data['M'])
    
    print(f"Processed data saved with name: {new_name}")

def sharpen_image(original_name, new_name, radius = 1, amount = 1):
    # 读取指定名字的数据
    original_data = find_image_data(original_name)
    
    if not original_data:
        print("Original data not found.")
        return

    # 应用锐化
    Y_blurred = unsharp_mask_multiband(original_data['Y'], radius, amount)
    label_blurred = unsharp_mask_multiband(original_data['label'], radius, amount)
    
    # 使用锐化后的Y和label数据，以及原始的其他数据，保存到数据库中新的记录
    insert_image_data(new_name, Y_blurred, label_blurred, original_data['M1'], original_data['M'])
    
    print(f"Processed data saved with name: {new_name}")

