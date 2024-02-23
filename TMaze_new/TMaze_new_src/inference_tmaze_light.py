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

# from RMDT_model.RMDT import mem_transformer_v2
from RATE_GTrXL import mem_transformer_v2_GTrXL
# from RATE_GTrXL import mem_transformer_v2_GTrXL
from TMaze_new.TMaze_new_src.tmaze_new_dataset import TMaze_data_generator, CombinedDataLoader
from TMaze_new.TMaze_new_src.val_tmaze import get_returns_TMaze
from TMaze_new.TMaze_new_src.additional import plot_cringe
from TMaze_new.TMaze_new_src.tmaze import TMazeClassicPassive

seeds_list = [0,2,3,4,6,9,13,15,18,24,25,31,3,40,41,42,43,44,48,49,50,
              51,62,63,64,65,66,69,70,72,73,74,75,83,84,85,86,87,88,91,
              92,95,96,97,98,100,102,105,106,107,
              1,5,7,8,10,11,12,14,16,
              17,19,20,21,22,23,26,27,28,29,30,32,34,35,36,37,38,39,45,
              46,47,52,53,54,55,56,57,58,59,60,61,67,68,71,76,77,78,79,80,81,82]

# Создание парсера аргументов
parser = argparse.ArgumentParser(description='Description of your program')
# Добавление аргументов
parser.add_argument('--model_mode', type=str, default='RMDT', help='Description of model_name argument')
parser.add_argument('--max_n_final', type=int, default=3, help='Description of max_n_final argument')
parser.add_argument('--ckpt_name', type=str, default='checkpoint_name', help='Description of name argument')
parser.add_argument('--ckpt_chooser', type=int, default=0, help='0 if last else int')
parser.add_argument('--ckpt_folder', type=str, default='', help='0 if last else int')
parser.add_argument('--arch_mode', type=str, default='TrXL', help='Description of model_name argument')

# Парсинг аргументов командной строки
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
    "n_layer": 3, # 6  # 8 
    "n_head": 1, # 4 # 10
    "d_model": 128, # 64                # 256
    "d_head": 128, # 32 # divider of d_model   # 128
    "d_inner": 128, # 128 # > d_model    # 512
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
if config["model_mode"] == "RMDT": 
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
elif config["model_mode"] == "RMDTM":
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

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # it's ruine np.random.choice()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

@torch.no_grad()
def sample(model, x, block_size, steps, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        
        if saved_context is not None:
            results = model(x_cond, actions, rtgs, None, timestep, *saved_context, mem_tokens=mem_tokens)
        else:
            results = model(x_cond, actions, rtgs, None, timestep, mem_tokens=mem_tokens) 
        # attn_map = results[1]
        #print(attn_map.shape) # num_mem_tokens x 1 x emb_dim
        logits = results[0][0][:,-1,:]
        mem_tokens = results[1]
        memory = results[0][2:]
        attn_map = model.attn_map
        
    return logits, mem_tokens, memory, attn_map

def get_returns_TMaze(model, ret, seed, episode_timeout, corridor_length, context_length, device, act_dim, config, create_video=False):
    #set_seed(seed)
    
    scale = 1
    channels = 5
    max_ep_len = episode_timeout

    env = TMazeClassicPassive(episode_length=episode_timeout, corridor_length=corridor_length, penalty=0, seed=seed, goal_reward=ret)
    state = env.reset() # {x, y, hint}
    np.random.seed(seed)
    where_i = state[0]
    mem_state = state[2]
    mem_state2 = state

    state = np.concatenate((state, np.array([0]))) # {x, y, hint, flag}
    state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)]))) # {x, y, hint, flag, noise}

    if create_video == True:
        print("down, required act: 3" if mem_state == -1.0 else "up,  required act: 1")

    state = torch.tensor(state).reshape(1, 1, channels)
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True
    Flag = 0
    frames = []
    HISTORY_LEN = context_length#context_length
    
    rews = []
    attentions = []
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    sim_states = []
    episode_return, episode_length = 0, 0

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    saved_context = None
    segment = 0
    prompt_steps = 0# 5
    act = None
    act_list= []

    switcher = False
    saved_mem = None
    

    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        act_new_segment = False
        if config["model_mode"] != 'DT':
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[-1:,:]
                    states = states[:, -1:, :]
                    target_return = target_return[:,-1:]
                    
                if t%(context_length)==0:# and t > context_length:
                    # print(states)
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        #out = new_notes[0] if new_notes is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                        
                    mem_tokens = new_mem
                    # !!!
                    #mem_tokens = saved_mem
                    saved_context = new_notes
                
        else:
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[1:,:]
                    states = states[:, 1:, :]
                    target_return = target_return[:,1:]
                    
                if t%(context_length)==0:# and t > context_length:
                    # print(states)
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        #out = new_notes[0] if new_notes is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                    mem_tokens = new_mem
                    saved_context = new_notes
                
        if t==0:
            act_to_pass = None
        else:
            act_to_pass = actions.unsqueeze(0)[:, 1:, :]
            if act_to_pass.shape[1] == 0:
                act_to_pass = None 
        
        sampled_action, new_mem, new_notes, attn_map = sample(model=model,  
                                                        x=states[:, :, 1:],
                                                        block_size=HISTORY_LEN, 
                                                        steps=1, 
                                                        sample=True, 
                                                        actions=act_to_pass, 
                                                        rtgs=target_return.unsqueeze(-1), 
                                                        mem_tokens=mem_tokens, #+torch.randn_like(mem_tokens), # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                        saved_context=saved_context)
        
        # !!!!!!!
        if t > 0 and t % (context_length-1) == 0 and switcher == False:
            switcher = True
            saved_mem = new_mem

        
        #act = np.random.choice([0, 1, 2, 3], p=torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        act = torch.argmax(torch.softmax(sampled_action, dim=-1).squeeze()).item()
        if create_video:
            print(t, "act", act, np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(), 3), "\tstate:", int(where_i), states[:, -1:, :].detach().cpu().numpy())
        actions[-1, :] = act
        act_list.append(act)
        state, reward, done, info = env.step(act)
        # print(t, env.time_step, env.x, env.y)
        
        #print(reward, done)
        
        ################################################################################################### TEN OF HINTS
        if t < config["hint_steps"]-1:
            state[2] = mem_state2[2]
        ################################################################################################################
        
         # {x, y, hint} -> {x, y, hint, flag}
        if state[0] != env.corridor_length:
            state = np.concatenate((state, np.array([0])))
        else:
            if Flag != 1:
                state = np.concatenate((state, np.array([1])))
                Flag = 1
            else:
                state = np.concatenate((state, np.array([0])))
                
        # {x, y, hint, flag} -> {x, y, hint, flag, noise}
        state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)])))
        
        
        delta_t = env.time_step - env.corridor_length - 1
        where_i = state[0]
        state = state.reshape(1, 1, channels)
        out_states.append(state)
        
        rews.append(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[0,-1] - (reward/scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        episode_return += reward
        episode_length += 1
        
        if (t+1) % (context_length) == 0 and t > 0:
            attentions.append(attn_map)
            
        if done:
            if create_video == True:
                if np.round(where_i, 4) == np.round(corridor_length, 4):
                    print("Junction achieved 😀 ✅✅✅")
                    print("Chosen act:", "up" if act == 1 else "down" if act == 3 else "wrong")
                    if mem_state == -1 and act == 3:
                        print("Correct choice 😀 ✅✅✅")
                    elif mem_state == 1 and act == 1:
                        print("Correct choice 😀 ✅✅✅")
                    else:
                        print("Wrong choice 😭 ⛔️⛔️⛔️")
                else:
                    print("Junction is not achieved 😭 ⛔️⛔️⛔️")
                
                print(f"{np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(),2)}")
            break  
    if create_video == True:
        print(f"Final position: [{int(where_i)}, {int(np.round(states.squeeze()[-1].tolist()[0 if channels == 3 else 1]))}] / [{int(corridor_length)}, {int(mem_state)}]")
        print("\n")
        
    return reward, act_list, t, np.array(out_states).squeeze(), delta_t, attentions


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

test = pd.read_csv('TMaze_new/TMaze_new_src/table_empty.csv')
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

