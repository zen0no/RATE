import os
import datetime
import torch
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import argparse
import yaml
import pickle
from torch.utils.data import random_split, DataLoader



import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from VizDoom.VizDoom_src.train import train
from TMaze_new.TMaze_new_src.utils import set_seed, get_intro_vizdoom
from VizDoom.VizDoom_src.utils.get_vizdoom_dataset import get_dataset

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
os.environ["OMP_NUM_THREADS"] = "1" 


with open("VizDoom/VizDoom_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# python3 VizDoom/VizDoom_src/train_vizdoom.py --model_mode 'RATE' --arch_mode 'TrXL' --ckpt_folder 'trash'

def create_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--model_mode',     type=str, default='RATE',  help='Model training mode. Available variants: "DT, DTXL, RATE (Ours), RATEM (RMT)"')    
    parser.add_argument('--arch_mode',      type=str, default='TrXL',  help='Model architecture mode. Available variants: "TrXL", "TrXL-I", "GTrXL"')
    parser.add_argument('--start_seed',     type=int, default=1,       help='Start seed')
    parser.add_argument('--end_seed',       type=int, default=3,       help='End seed')
    parser.add_argument('--ckpt_folder',    type=str, default='ckpt',  help='Checkpoints directory')

    return parser



#================================================== DATA LOADING =============================================================#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! change file !!!!!!!!!!!!!!!!

if __name__ == '__main__':
    get_intro_vizdoom()
    
    args = create_args().parse_args()

    train_pickle_file = 'VizDoom/VizDoom_data/val_VizDoom_Two_Colors_Column_disappear_delay_45_no_walls_agent_p1_01.pickle'
    DATA2_train = []
    with open(train_pickle_file, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                DATA2_train.append(data)
            except EOFError:
                break
    DATA_train = {}
    for key in tqdm(DATA2_train[0].keys()):
        DATA_train[key] = [d[key] for d in DATA2_train]
    ############################################################################

    model_mode = args.model_mode
    start_seed = args.start_seed
    end_seed = args.end_seed
    arch_mode = args.arch_mode
    ckpt_folder = args.ckpt_folder

    config["model_mode"] = model_mode
    config["arctitecture_mode"] = arch_mode
    config["training_config"]["sections"] = 3

    # RUN = 1
    for RUN in range(start_seed, end_seed+1):
        set_seed(RUN)

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


        TEXT_DESCRIPTION = ""
        mini_text = f"arch_mode_{config['arctitecture_mode']}"
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S").replace('-', '_')
        group = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}'
        name = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}_RUN_{RUN}_{date_time}'
        current_dir = os.getcwd()
        current_folder = os.path.basename(current_dir)
        ckpt_path = f'../{current_folder}/VizDoom/VizDoom_checkpoints/{ckpt_folder}/{name}/'
        isExist = os.path.exists(ckpt_path)
        if not isExist:
            os.makedirs(ckpt_path)

        if config["wandb_config"]["wwandb"]:
            #run = wandb.init(entity="rmdt", project="VizDoom_clear", name=name, group=group, config=config, save_code=True, reinit=True)
            run = wandb.init(project=config['wandb_config']['project_name'], name=name, group=group, config=config, save_code=True, reinit=True)

        #================================================== DATALOADERS CREATION ======================================================#
        train_dataset = get_dataset(DATA_train, 
                                    gamma=config["data_config"]["gamma"], 
                                    max_length=config["training_config"]["sections"]*config["training_config"]["context_length"], 
                                    normalize=True)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=config["training_config"]["batch_size"], 
                                      shuffle=True, 
                                      num_workers=4)

        print(f"Train: {len(train_dataset)}")
        #==============================================================================================================================#
        wandb_step = 0
        model = train(ckpt_path, config, train_dataloader)
                
        if config["wandb_config"]["wwandb"]:
            run.finish()