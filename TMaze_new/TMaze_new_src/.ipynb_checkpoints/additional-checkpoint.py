import os
OMP_NUM_THREADS = '1'
os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS

import sys
sys.path.append("RATE_model/RATE/")
import mem_transformer_v2

import datetime
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import glob


sys.path.append("TMaze/TMaze_curriculum/TMaze_curriculum_src/")
from PIL import Image

# sns.set_style("whitegrid")
# sns.set_palette("colorblind")
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

def get_flatten_pic_attn(attn_map_list):
    # Pad the smaller array with zeros (or any other desired values)
    # size_diff = len(attn_map_list[0]) - len(attn_map_list[1])
    # if size_diff > 0:
    #     attn_map_list[1] = np.pad(attn_map_list[1], ((0, size_diff), (0, 0)), mode='constant')
    # elif size_diff < 0:
    #     attn_map_list[0] = np.pad(attn_map_list[0], ((0, abs(size_diff)), (0, 0)), mode='constant')
    # if len(attn_map_list) == 1:
    #     print(attn_map_list[0].shape)
    # else:
    #     print(attn_map_list[0].shape, attn_map_list[1].shape)
        
    ans = np.concatenate(attn_map_list, axis=1) if len(attn_map_list) > 1 else attn_map_list[0]
    
    return ans

def get_flatten_pic_attn_state(attn_map_list, config, mem_at_end):
    # Pad the smaller array with zeros (or any other desired values)
    # size_diff = len(attn_map_list[0]) - len(attn_map_list[1])
    # if size_diff > 0:
    #     attn_map_list[1] = np.pad(attn_map_list[1], ((0, size_diff), (0, 0)), mode='constant')
    # elif size_diff < 0:
    #     attn_map_list[0] = np.pad(attn_map_list[0], ((0, abs(size_diff)), (0, 0)), mode='constant')
    
    if config["model_mode"] == "RATE":
        lst = []
        for i in range(len(attn_map_list)):
            attn_weights = attn_map_list[i]

            nmt = config["num_mem_tokens"]
            if mem_at_end == True:
                attn_weights2 = attn_weights[nmt:-nmt, nmt:-nmt][1::3, 1::3]   #rtg 0::3 state 1::3 action 2::3
            else:
                attn_weights2 = attn_weights[nmt:, nmt:][1::3, 1::3]
            len1 = attn_weights2.shape[0]
            new_attn_map = np.zeros(((nmt)*2+len1, nmt*2+len1))
            new_attn_map[:nmt, :nmt] = attn_weights[:nmt, :nmt]
            new_attn_map[nmt:-nmt, :nmt] = attn_weights[nmt:-nmt, :nmt][1::3, :]
            new_attn_map[-nmt:, nmt:-nmt] = attn_weights[-nmt:, nmt:-nmt][:, 1::3]
            new_attn_map[-nmt:, -nmt:] = attn_weights[-nmt:, -nmt:]
            new_attn_map[nmt:-nmt, nmt:-nmt] = attn_weights2

            lst.append(new_attn_map)
            
    elif config["model_mode"] == "DT":
        lst = []
        for i in range(len(attn_map_list)):
            attn_weights = attn_map_list[i]
            if mem_at_end == True:
                attn_weights2 = attn_weights[1::3, 1::3]   #rtg 0::3 state 1::3 action 2::3
            else:
                attn_weights2 = attn_weights[1::3, 1::3]
            len1 = attn_weights2.shape[0]
            new_attn_map = np.zeros((len1, len1))
            new_attn_map = attn_weights
            new_attn_map = attn_weights[1::3, :]
            new_attn_map = attn_weights[:, 1::3]
            new_attn_map = attn_weights
            new_attn_map = attn_weights2

            lst.append(new_attn_map)        
        
    return np.concatenate(lst, axis=1)

def draw_number(num_str):
    num = int(num_str)
    assert 10 <= num <= 999, "Number must be between 10 and 999"
    
    # Create a 20x20 array of zeros
    arr = np.zeros((5, 15))
    
    # Calculate the width and height of each digit
    width = DIGITS[0].shape[0]
    height = DIGITS[0].shape[1]
    
    # Loop through each digit in the number
    for i, digit in enumerate(num_str):
        # Calculate the x and y coordinates of the top-left corner of the digit
        x = 0
        y = i * width
        
        # Get the corresponding digit image
        digit_arr = DIGITS[int(digit)]
        
        # Resize the digit image to the correct size
        #digit_arr = np.repeat(np.repeat(digit_arr, width, axis=1), height, axis=0)
        
        # Insert the digit image into the array
        arr[x:x+width, y:y+height] = digit_arr
    
    return arr

# Define the digit images
DIGITS = [
    # 0
    np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ]),
    # 1
    np.array([
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 1]
    ]),
    # 2
    np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1]
    ]),
    # 3
    np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0]
    ]),
    # 4
    np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
    ]),
    # 5
    np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 0]
    ]),
    # 6
    np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ]),
    # 7
    np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ]),
    # 8
    np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ]),
    # 9
    np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])
]

def stretch_array(array, target_shape):
    # Проверяем входные данные
    if not isinstance(array, np.ndarray):
        raise ValueError("Переданный объект не является массивом NumPy")

    if len(array.shape) != len(target_shape):
        raise ValueError("Число измерений не совпадает")

    scale_factor = [target_dim // src_dim for target_dim, src_dim in zip(target_shape, array.shape)]
    stretched_array = np.repeat(array, scale_factor[0], axis=0)
    stretched_array = np.repeat(stretched_array, scale_factor[1], axis=1)

    return stretched_array

# pil_image = Image.open('../RATE/TMaze/TMaze_curriculum/anime.png')
# resized_image = pil_image.resize((40, 40))


def plot_cringe(attn_map, junction_pos, config, mem_at_end):
    A, B = get_flatten_pic_attn(attn_map), get_flatten_pic_attn_state(attn_map, config, mem_at_end)
    num_str = str(junction_pos)
    arr = draw_number(num_str)
    target_shape = (20, 50)
    stretched_array = stretch_array(arr, target_shape)

    C = np.zeros((A.shape[0] + B.shape[0] + 1, A.shape[1]))
    C[:A.shape[0], :A.shape[1]] = A
    C[A.shape[0], 0:] = 1
    C[A.shape[0]+1:, :B.shape[1]] = B
    #C[A.shape[0]+1:, B.shape[1]:B.shape[1]+40] = np.array(resized_image).sum(axis=-1).astype(np.float32) / 255.
    if A.shape[1] - B.shape[1] - 40 < stretched_array.shape[1]:
        C[A.shape[0]+1+20:A.shape[0]+1+5+20, B.shape[1]+40:B.shape[1]+40+15] = arr # stretched_array
    else:
        C[A.shape[0]+1+10:A.shape[0]+1+10+20, B.shape[1]+40:B.shape[1]+40+45] = stretched_array
        
#     target_shape = (12, 24)
#     stretched_array = stretch_array(arr, target_shape)

#     C = np.zeros((A.shape[0] + B.shape[0] + 1, A.shape[1]))
#     C[:A.shape[0], :A.shape[1]] = A
#     C[A.shape[0], 0:] = 1
#     C[A.shape[0]+1:, :B.shape[1]] = B
#     #C[A.shape[0]+1:, B.shape[1]:B.shape[1]+40] = np.array(resized_image).sum(axis=-1).astype(np.float32) / 255.
#     if A.shape[1] - B.shape[1] - 40 < stretched_array.shape[1]:
#         C[A.shape[0]+1+12:A.shape[0]+1+5+12, B.shape[1]:B.shape[1]+15] = arr # stretched_array
#     else:
#         C[A.shape[0]+1+10:A.shape[0]+1+10+12, B.shape[1]+40:B.shape[1]+40+45] = stretched_array

    # plt.figure(figsize=(10,10))
    # plt.imshow(C, cmap="magma")
    # plt.show()
    
    return C