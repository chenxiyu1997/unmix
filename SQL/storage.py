import sqlite3
import numpy as np
import scipy.io as sio

# 初始化数据库连接
conn = sqlite3.connect('hsi_data.db')
c = conn.cursor()

def create_tables():
    # 创建表，每个表都有id, name和data列
    tables = ['images', 'labels', 'lidar', 'M_init', 'M_true']
    for table in tables:
        c.execute(f'''CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY,
            name TEXT,
            data BLOB
        )''')
    conn.commit()

def insert_data(table_name, name, data):
    """插入数据到数据库"""
    data_blob = sqlite3.Binary(data.tobytes())  # 将numpy数组转换为二进制
    c.execute(f"INSERT INTO {table_name} (name, data) VALUES (?, ?)", (name, data_blob))
    conn.commit()
    return c.lastrowid  # 返回新插入行的ID

def get_data_list(table_name):
    """获取指定表中所有数据的列表"""
    c.execute(f"SELECT id, name FROM {table_name}")
    return c.fetchall()

def get_data_by_id(table_name, record_id):
    """根据ID获取数据"""
    c.execute(f"SELECT name, data FROM {table_name} WHERE id = ?", (record_id,))
    result = c.fetchone()
    if result:
        name, data_blob = result
        data = np.frombuffer(data_blob, dtype=np.float32)  # 假设数据类型为np.float32
        return name, data
    else:
        return None

# 确保表已创建
create_tables()


image_file = r'../data/muffle_dataset_130_90.mat'
# num_classes = 5
# band = 64
# col = 90
# row = 130


input_data = sio.loadmat(image_file)
image = input_data['Y']
image = image.astype(np.float32)
label = input_data['label']
label = label.astype(np.float32).transpose(2, 1, 0)
lidar = input_data['MPN']
lidar = lidar.astype(np.float32)
M_init = input_data['M1']
# M_init = torch.from_numpy(M_init).unsqueeze(2).unsqueeze(3).float()
M_true = input_data['M']

# 示例：插入数据
# 假设 image, label, lidar, M_init, M_true 是已经加载的数据
new_id = insert_data('images', 'Image Name', image)
print(f"新插入的记录ID: {new_id}")

# 示例：获取数据列表
# data_list = get_data_list('images')
# for id, name in data_list:
#     print(f"ID: {id}, Name: {name}")
print('1111111111111执行了')
# 示例：根据ID获取数据
# record_id = 1  # 示例ID
# name, data = get_data_by_id('images', record_id)
# print(f"Name: {name}, Data: {data[:10]}")

# 关闭数据库连接
conn.close()