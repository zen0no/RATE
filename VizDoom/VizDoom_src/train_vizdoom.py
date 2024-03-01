import os
import datetime
import torch
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import pickle
from torch.utils.data import random_split, DataLoader

OMP_NUM_THREADS = '1'
os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS 

# from RMDT_model.RMDT import mem_transformer_v2
# from RATE_GTrXL import mem_transformer_v2_GTrXL
from RATE_GTrXL import mem_transformer_v2_GTrXL_3
from VizDoom.VizDoom_src.get_vizdoom_dataset import get_dataset

#================================================== DATA LOADING =============================================================#
train_pickle_file = 'VizDoom/VizDoom_data/train_VizDoom_Two_Colors_Column_disappear_delay_45_no_walls_agent_p1_01.pickle'
val_pickle_file = 'VizDoom/VizDoom_data/val_VizDoom_Two_Colors_Column_disappear_delay_45_no_walls_agent_p1_01.pickle'
DATA2_train, DATA2_val = [], []
with open(train_pickle_file, 'rb') as f:
    while True:
        try:
            data = pickle.load(f)
            DATA2_train.append(data)
        except EOFError:
            break
with open(val_pickle_file, 'rb') as f:
    while True:
        try:
            data = pickle.load(f)
            DATA2_val.append(data)
        except EOFError:
            break
DATA_train, DATA_val = {}, {}
for key in tqdm(DATA2_train[0].keys()):
    DATA_train[key] = [d[key] for d in DATA2_train]
    
for key in tqdm(DATA2_val[0].keys()):
    DATA_val[key] = [d[key] for d in DATA2_val]
#==============================================================================================================================#

#===================================================== TRAIN FUNCTION =========================================================#
def train(ckpt_path, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the config dictionary to initialize the model
    model = mem_transformer_v2_GTrXL_3.MemTransformerLM(
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
    )

    torch.nn.init.xavier_uniform_(model.r_w_bias);
    torch.nn.init.xavier_uniform_(model.r_r_bias);
    wandb_step  = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],  weight_decay=config["weight_decay"], betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/config["warmup_steps"], 1))
    raw_model = model.module if hasattr(model, "module") else model
        
    model.to(device)
    model.train()
    
    wwandb = config["wwandb"]
    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    best_val_loss = np.inf
    epochs_without_improvement = 0
    max_epochs_without_improvement = 5000
    suc_rate, ep_time = 0, 0
    
    learning_rate_decay = 0.99
    switch = False
    block_size = 3*config["context_length"]
    EFFECTIVE_SIZE_BLOCKS = config["context_length"] * config["sections"]
    BLOCKS_CONTEXT = block_size//3
    
    pbar = tqdm(range(config["epochs"]))
    for epoch in pbar:
        train_imgs = []
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch
            #print('1', s.shape)
            memory = None
            mem_tokens=None
            
            if config["model_mode"] == 'DT':
                block_part_range = range(1)
            else:
                block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
                
            for block_part in block_part_range:
                if config["model_mode"] == 'DT':
                    x1 = s.to(device)
                    y1 = a.to(device).float()
                    r1 = rtg.to(device).float()
                    t1 = timesteps.to(device)
                    masks1 = masks.to(device)
                else:
                    from_idx = block_part*(BLOCKS_CONTEXT)
                    to_idx = (block_part+1)*(BLOCKS_CONTEXT)
                    x1 = s[:, from_idx:to_idx, :].to(device)
                    y1 = a[:, from_idx:to_idx, :].to(device).float()
                    r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(device).float() 
                    t1 = timesteps[:, from_idx:to_idx].to(device)
                    masks1 = masks[:, from_idx:to_idx].to(device)
                    
                #print('2', x1.shape)
                model.flag = 1 if block_part == list(range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT))[-1] else 0
                if mem_tokens is not None:
                    mem_tokens = mem_tokens.detach()
                elif raw_model.mem_tokens is not None:
                    mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)
                with torch.set_grad_enabled(is_train):
                    optimizer.zero_grad()
                        
                    res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None \
                    else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                    memory = res[0][2:]
                    logits, train_loss = res[0][0], res[0][1]
                    mem_tokens = res[1]
                    #train_imgs.append(model.attn_map)
                
                    if wwandb:
                        wandb.log({"train_loss":  train_loss.item()})

                if is_train:
                    model.zero_grad()
                    optimizer.zero_grad()
                    train_loss.backward(retain_graph=True)
                    if config["grad_norm_clip"] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_norm_clip"])
                    optimizer.step()
                    scheduler.step()
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
            it_counter += 1 
            
        #if model.flag == 1:
        pbar.set_description(f"ep {epoch+1} it {it} tTotal {train_loss.item():.2f} lr {lr:e}")     
        
        # Scheduler changer
        if it_counter >= config["warmup_steps"] and switch == False:
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=learning_rate_decay, patience=patience, mode="min")
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001,
                                                          total_iters=5*len(train_dataloader)*config["epochs"])
            switch = True
        
        # if wwandb:
        #     if model.flag == 1:
        #         image = np.concatenate(train_imgs[-(list(range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT))[-1]+1):], axis=1)
        #         images = wandb.Image(image, caption="Train attention map for the 0 batch and 0 head")
        #         wandb.log({"train_attention_maps": images})
        #         train_imgs = []
                
        #         image = np.concatenate(val_imgs[-(list(range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT))[-1]+1):], axis=1)
        #         images = wandb.Image(image, caption="Val attention map for the 0 batch and 0 head")
        #         wandb.log({"val_attention_maps": images})
        #         val_imgs = []
        
        # Save
        if (epoch + 1) % 250 == 0 or epoch == config["epochs"] - 1:
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": wandb_step})
            #torch.save(model.state_dict(), ckpt_path + '_' + str(wandb_step) + '_KTD.pth')
            torch.save(model.state_dict(), ckpt_path + '_' + str(wandb_step) + '_KTD.pth')
            
    return model
#==============================================================================================================================#
# RUN = 1
for RUN in range(1, 3+1):
    config = {
        "batch_size": 128, # 128
        "warmup_steps": 50, 
        "grad_norm_clip": 1.0, # 0.25 
        "wwandb": True, 
        "sections": 3,            # 3     #####################
        "context_length": 30,
        "epochs": 1000, #250
        "mode": "doom",
        "model_mode": "DT",
        "state_dim": 3,
        "act_dim": 1,
        "vocab_size": 10000,
        "n_layer": 3, # 6 # 4
        "n_head": 1, #  8 # 2
        "d_model": 128, # 64                # 256
        "d_head": 128, # 32 # divider of d_model   # 128
        "d_inner": 128, # 128 # > d_model    # 512
        "dropout": 0.2, # 0.05
        "dropatt": 0.05, # 0.05
        "MEM_LEN": 2,
        "ext_len": 1,
        "tie_weight": False,
        "num_mem_tokens": 5, # 5
        "mem_at_end": True,
        "learning_rate": 1e-3, # 1e-4
        "weight_decay": 0.1,
        "gamma": 1.0,
    }

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

    TEXT_DESCRIPTION = "fixed"##_bigger"#"loss_all"
    mini_text = "TrXL_qwe_false"#"no_curr_data_1_9_inf_on_9_segments"
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S").replace('-', '_')
    group = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}'
    name = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}_RUN_{RUN}_{date_time}'
    current_dir = os.getcwd()
    current_folder = os.path.basename(current_dir)
    ckpt_path = f'../{current_folder}/VizDoom/VizDoom_checkpoints/VizDoom_TrXL_qwe_false/{name}/'
    isExist = os.path.exists(ckpt_path)
    if not isExist:
        os.makedirs(ckpt_path)

    if config["wwandb"]:
        #run = wandb.init(entity="rmdt", project="VizDoom_clear", name=name, group=group, config=config, save_code=True, reinit=True)
        run = wandb.init(project="VizDoom_GTrXL", name=name, group=group, config=config, save_code=True, reinit=True)

    #================================================== DATALOADERS CREATION ======================================================#
    train_dataset = get_dataset(DATA_train, gamma=config["gamma"], max_length=config["sections"]*config["context_length"], normalize=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_dataset = get_dataset(DATA_val, gamma=config["gamma"], max_length=config["sections"]*config["context_length"], normalize=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    #==============================================================================================================================#
    wandb_step = 0
    model = train(ckpt_path, config)
            
    if config["wwandb"]:
        run.finish()