import numpy as np
import yaml
import torch
from tqdm import tqdm

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from RATE_GTrXL import mem_transformer_v2_GTrXL
from VizDoom.VizDoom_src.inference.val_vizdoom import get_returns_VizDoom
from TMaze_new.TMaze_new_src.utils import seeds_list, get_intro2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# python3 VizDoom/VizDoom_src/inference/inference_vizdoom.py

with open("VizDoom/VizDoom_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    get_intro2()

    episode_timeout = 2100
    use_argmax = False
    # 1024: mean and std: tensor([13.6313, 19.6772, 14.7505]) tensor([16.7388, 20.3475, 10.3455])
    MEAN = torch.tensor([13.5173, 19.6073, 14.7196])
    STD = torch.tensor([16.2992, 20.0957, 10.2147])

    config["model_mode"] = 'RATE'
    config["arctitecture_mode"] = 'TrXL'
    config["training_config"]["sections"] = 3

    """ ARCHITECTURE MODE """
    if config["arctitecture_mode"] == "TrXL":
        config["model_config"]["use_gate"] = False
        config["model_config"]["use_stable_version"] = False
    elif config["arctitecture_mode"] == "TrXL-I":
        config["model_config"]["use_gate"] = False
        config["model_config"]["use_stable_version"] = True
    elif config["arctitecture_mode"] == "GTrXL":
        config["model_config"]["use_gate"] = True
        config["model_config"]["use_stable_version"] = True     

    print(f"Selected Architecture: {config['arctitecture_mode']}")  

    """ MODEL MODE """
    if config["model_mode"] == "RATE": 
        config["model_config"]["mem_len"] = 2 ########################### 2 FOR DTXL 0
        config["model_config"]["mem_at_end"] = True ########################### True FOR DTXL False
    elif config["model_mode"] == "DT":
        config["model_config"]["mem_len"] = 0 ########################### 2 FOR DTXL 0
        config["model_config"]["mem_at_end"] = False ########################### True FOR DTXL False
        config["model_config"]["num_mem_tokens"] = 0
    elif config["model_mode"] == "DTXL":
        config["model_config"]["mem_len"] = 2
        config["model_config"]["mem_at_end"] = False
        config["model_config"]["num_mem_tokens"] = 0
    elif config["model_mode"] == "RATEM":
        config["model_config"]["mem_len"] = 0
        config["model_config"]["mem_at_end"] = True

    print(f"Selected Model: {config['model_mode']}")

    model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

    ckpt_path = f'VizDoom/VizDoom_checkpoints/RATE/good_z_norm_arch_mode_TrXL_RATE_RUN_1_2024_03_01_23_23_20/_8_KTD.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    _ = model.eval()


    for ret in [1.01, 3.9, 11.1, 56.5]:
        print("TARGET RETURN:", ret)
        goods, bads = 0, 0
        acts = []
        pbar = tqdm(range(len(seeds_list)))
        returns = []
        ts = []
        for i in pbar:
            episode_return, act_list, t, _, _ = get_returns_VizDoom(model=model, ret=ret, seed=seeds_list[i], 
                                                                    episode_timeout=episode_timeout, 
                                                                    context_length=config["training_config"]["context_length"], 
                                                                    device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                    config=config,
                                                                    mean=MEAN,
                                                                    std=STD,
                                                                    use_argmax=use_argmax, create_video=False)
            acts += act_list
            returns.append(episode_return)
            ts.append(t)
            pbar.set_description(f"Time: {t}, Return: {episode_return:.2f}")
            
        print(f"Mean reward: {np.mean(returns):.2f}")
        print(f"STD reward: {np.std(returns):.2f}")
        print(f"Mean T: {np.mean(ts):.2f}")
        print(f"STD T: {np.std(ts):.2f}")