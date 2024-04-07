import numpy as np
import itertools

def compute_rmse(x_true, x_pre):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt( ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c) )

def compute_sad(x_true, x_pre):
    band, pixel = x_true.shape
    result = np.linalg.norm(x_true, ord=2, axis=0) * np.linalg.norm(x_pre, ord=2, axis=0)
    sad = np.arccos( (x_true * x_pre).sum(axis=0)/result )
    return sad.mean()

def close_position(x_true, x_pre):
    min_sad = np.inf
    best_position = None

    # 暴力搜索所有可能的列组合
    for position in itertools.permutations(range(x_pre.shape[1])):
        current_sad = compute_sad(x_true, x_pre[:, position])
        if current_sad < min_sad:
            min_sad = current_sad
            best_position = position

    print("position: ", best_position)
    return best_position
    
    
    