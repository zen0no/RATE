import os
import datetime
import torch
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import sys
import matplotlib.pyplot as plt
import random
import argparse
import pandas as pd
import re

OMP_NUM_THREADS = '1'
os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS 

# from RATE_model.RATE import mem_transformer_v2
from RATE_GTrXL import mem_transformer_v2_GTrXL
# from RATE_GTrXL import mem_transformer_v2_GTrXL
from TMaze_new.TMaze_new_src.utils.tmaze_new_dataset import TMaze_data_generator, CombinedDataLoader
from TMaze_new.TMaze_new_src.utils.additional2 import plot_cringe 
from TMaze_new.TMaze_new_src.inference.val_tmaze import get_returns_TMaze


from TMaze_new.TMaze_new_src.utils import seeds_list

parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('--model_mode', type=str, default='RATE', help='Description of model_name argument')
parser.add_argument('--max_n_final', type=int, default=3, help='Description of max_n_final argument')
parser.add_argument('--ckpt_name', type=str, default='checkpoint_name', help='Description of name argument')
parser.add_argument('--ckpt_chooser', type=int, default=0, help='0 if last else int')
parser.add_argument('--ckpt_folder', type=str, default='', help='0 if last else int')
parser.add_argument('--arch_mode', type=str, default='TrXL', help='Description of model_name argument')

args = parser.parse_args()
model_mode = args.model_mode
min_n_final = 1
max_n_final = args.max_n_final
ckpt_name = args.ckpt_name
ckpt_chooser = args.ckpt_chooser
ckpt_folder = args.ckpt_folder
arch_mode = args.arch_mode

config = {
    # data parameters
    "max_segments": max_n_final,
    "multiplier": 50,
    "hint_steps": 1,

    # "episode_timeout": max_n_final*30,
    # "corridor_length": max_n_final*30-2, # 58
    "episode_timeout": 5*30,
    "corridor_length": 5*30-2, # 58
    
    "cut_dataset": False,

    "batch_size": 64, # 64
    "warmup_steps": 50,  # 100
    "grad_norm_clip": 1.0, # 0.25 
    "wwandb": True, 
    "sections": max_n_final,                  ##################### d_head * nmt = diff params
    "context_length": 30,
    "epochs": 250, #250
    "mode": "tmaze",
    "model_mode": model_mode,
    "state_dim": 4,
    "act_dim": 1,
    "vocab_size": 10000,
    "n_layer": 8, # 6  # 8 
    "n_head": 10, # 4 # 10
    "d_model": 256, # 64                # 256
    "d_head": 128, # 32 # divider of d_model   # 128
    "d_inner": 512, # 128 # > d_model    # 512
    "dropout": 0.05, # 0.1  
    "dropatt": 0.00, # 0.05
    "MEM_LEN": 2,
    "ext_len": 1,
    "tie_weight": False,
    "num_mem_tokens": 5, # 5
    "mem_at_end": True,
    "coef": 0.0,
    "learning_rate": 1e-4, # 1e-4
    "weight_decay": 0.1,
    "curriculum": True,
    "use_erl_stop": True,
    "online_inference": False,
    "arctitecture_mode": arch_mode,

    # environment parameters
    "desired_reward": 4.0, # defalt: 1.0
    "win_only": False   # default: True
}

""" ARCHITECTURE MODE """
if config["arctitecture_mode"] == "TrXL":
    config["use_gate"] = False
    config["use_stable_version"] = False
elif config["arctitecture_mode"] == "TrXL-I":
    config["use_gate"] = False
    config["use_stable_version"] = True
elif config["arctitecture_mode"] == "GTrXL":
    config["use_gate"] = True
    config["use_stable_version"] = True     

print(f"Selected Architecture: {config['arctitecture_mode']}")  

""" MODEL MODE """
if config["model_mode"] == "RATE": 
    config["MEM_LEN"] = 2 ########################### 2 FOR DTXL 0
    config["mem_at_end"] = True ########################### True FOR DTXL False
elif config["model_mode"] == "DT":
    config["MEM_LEN"] = 0 ########################### 2 FOR DTXL 0
    config["mem_at_end"] = False ########################### True FOR DTXL False
    config["num_mem_tokens"] = 0
elif config["model_mode"] == "DTXL":
    config["MEM_LEN"] = 2
    config["mem_at_end"] = False
    config["num_mem_tokens"] = 0
elif config["model_mode"] == "RATEM":
    config["MEM_LEN"] = 0
    config["mem_at_end"] = True

print(f"Selected Model: {config['model_mode']}")  

model = mem_transformer_v2_GTrXL.MemTransformerLM(
    STATE_DIM=config["state_dim"],
    ACTION_DIM=config["act_dim"],
    n_token=config["vocab_size"],
    n_layer=config["n_layer"],
    n_head=config["n_head"],
    d_model=config["d_model"],
    d_head=config["d_head"],
    d_inner=config["d_inner"],
    dropout=config["dropout"],
    dropatt=config["dropatt"],
    mem_len=config["MEM_LEN"],
    ext_len=config["ext_len"],
    tie_weight=config["tie_weight"],
    num_mem_tokens=config["num_mem_tokens"],
    mem_at_end=config["mem_at_end"],
    mode=config["mode"],
    use_gate=config["use_gate"],
    use_stable_version=config["use_stable_version"]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.getcwd()
current_folder = os.path.basename(current_dir)
name = ckpt_name
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ckpt_folder = 'classic_1'
ckpt_path = f'../{current_folder}/TMaze_new/TMaze_new_checkpoints/{ckpt_folder}/{name}/'
folder_name = f'TMaze_new_inference_{ckpt_folder}'
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

folder_path = ckpt_path

files = os.listdir(folder_path)
files = [f for f in files if f.endswith('_KTD.pth') and '_' in f]#[f for f in files if f.endswith('_KTD.pth') and '_' in f]
last_file = files[-1]

if ckpt_chooser == 0:
    ckpt_num = last_file
else:
    ckpt_num = ckpt_chooser

model.load_state_dict(torch.load(ckpt_path + last_file, map_location=device), strict=True)
model.to(device)
print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
_ = model.eval()    




#print("NAME:", name)
print("Checkpoint:", ckpt_num)
print("NAME:", name)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# with open(f'../{current_folder}/TMaze_new/TMaze_new_inference/TMaze_new_inference_qkw_norm_false_TrXL_bs_64/{name}.txt', 'w') as f:
#     print("Checkpoint:", ckpt_num, file=f)
#     print("NAME:", name, file=f)
#     for segments in [1, 2, 3, 5, 7, 9, 12]:
#         rets = []
#         for seed in tqdm(seeds_list):
#             episode_timeout = 30*segments
#             corridor_length = 30*segments - 2
#             create_video = False

#             episode_return, act_list, t, states, _, attn_map = get_returns_TMaze(model=model, ret=1.0, seed=seed, 
#                                                                                 episode_timeout=episode_timeout, corridor_length=corridor_length, 
#                                                                                 context_length=config["context_length"], 
#                                                                                 device=device, act_dim=config["act_dim"], 
#                                                                                 config=config, create_video=create_video)
#             rets.append(episode_return)

#         print("SEGMENTS", segments, np.mean(rets), np.std(rets), sep='\t')
#         print("SEGMENTS", segments, np.mean(rets), np.std(rets), sep='\t', file=f)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

segmentss, means, stds = [], [], []
for segments in [1, 2, 3, 5, 7, 9, 12]:
    rets = []
    for seed in tqdm(seeds_list):
        episode_timeout = 30*segments
        corridor_length = 30*segments - 2
        create_video = False
 
        episode_return, act_list, t, states, _, attn_map = get_returns_TMaze(model=model, ret=config["desired_reward"], seed=seed, 
                                                                            episode_timeout=episode_timeout, corridor_length=corridor_length, 
                                                                            context_length=config["context_length"], 
                                                                            device=device, act_dim=config["act_dim"], 
                                                                            config=config, create_video=create_video)
        rets.append(int(episode_return == config["desired_reward"]))

    segmentss.append(segments)
    means.append(np.mean(rets))
    stds.append(np.std(rets))

    print("SEGMENTS", segments, np.mean(rets), np.std(rets), sep='\t')

test = pd.read_csv('TMaze_new/TMaze_new_src/utils/table_empty.csv')
match = re.search(r'RUN_(\d+)', name)
if match:
    run_num = match.group(1)

test.iloc[1, 0] = name
test.iloc[1, 1] = run_num
test.iloc[1, 2] = ckpt_num

for i in range(len(means)):
    test.iloc[1, 3+i*2] = means[i]
    test.iloc[1, 4+i*2] = stds[i]

string = name
string_no_data = string.split('_')[:-6]
new_string = ''
for el in string_no_data:
    new_string += el + "_"

new_string = new_string[:-1]

new_string_no_run = new_string.split('_')[:-2]

new_string2 = ''
for el in new_string_no_run:
    new_string2 += el + "_"

new_string2 = new_string2[:-1]

save_path = f'../{current_folder}/TMaze_new/TMaze_new_inference/{folder_name}/{new_string2}/'
isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)

test.to_csv(f'../{current_folder}/TMaze_new/TMaze_new_inference/{folder_name}/{new_string2}/{name}.csv', index=False)

