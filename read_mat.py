import scipy.io
import matplotlib.pyplot as plt

# 加载 .mat 文件
data = scipy.io.loadmat('./data/muffle_dataset_130_90.mat')

# 假设 DSM 数据在字典中的键为 'DSM'
dsm_data = data['DSM']

# 展示 DSM 数据
plt.imshow(dsm_data)
plt.colorbar()
plt.show()