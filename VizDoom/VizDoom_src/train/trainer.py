import numpy as np
import torch
import wandb
from tqdm import tqdm

from RATE_GTrXL import mem_transformer_v2_GTrXL
from VizDoom.VizDoom_src.utils import z_normalize, inverse_z_normalize
from VizDoom.VizDoom_src.inference.val_vizdoom import get_returns_VizDoom
from MemoryMaze.MemoryMaze_src.inference.val_mem_maze import get_returns_MemoryMaze
from TMaze_new.TMaze_new_src.utils import seeds_list

def train(ckpt_path, config, train_dataloader, mean, std, max_segments):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    episode_timeout = config["online_inference_config"]["episode_timeout"]
    use_argmax = config["online_inference_config"]["use_argmax"]
    # MEAN = torch.tensor([13.6313, 19.6772, 14.7505])
    # STD  = torch.tensor([16.7388, 20.3475, 10.3455])
    MEAN = mean
    STD = std

    model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

    # !!!!!!!!!!!!!!!!!!!!!!
    print(config)
    # !!!!!!!!!!!!!!!!!!!!!!
    
    torch.nn.init.xavier_uniform_(model.r_w_bias);
    torch.nn.init.xavier_uniform_(model.r_r_bias);
    wandb_step  = 0

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config["training_config"]["learning_rate"], 
                                  weight_decay=config["training_config"]["weight_decay"], 
                                  betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/config["training_config"]["warmup_steps"], 1))
    raw_model = model.module if hasattr(model, "module") else model
        
    model.to(device)
    model.train()
    
    wwandb = config["wandb_config"]["wwandb"]
    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    switch = False

    EFFECTIVE_SIZE_BLOCKS = config["training_config"]["context_length"] * config["training_config"]["sections"]
    BLOCKS_CONTEXT = config["training_config"]["context_length"]
    
    pbar = tqdm(range(config["training_config"]["epochs"]))
    for epoch in pbar:
        train_imgs = []
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch
            #print(s.shape, a.shape, rtg.shape, timesteps.shape, masks.shape)
            # ! NORMALIZATION
            if config["data_config"]["normalize"]:
                _b, _l, _c, _h, _w = s.shape
                s = s.reshape(_b*_l, _c, _h, _w)
                s = z_normalize(s, MEAN, STD)
                s = s.reshape(_b, _l, _c, _h, _w)
            # ! NORMALIZATION
            # print('1', s.shape)
            memory = None
            mem_tokens=None
            
            block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
            # print("RANGE", block_part_range)
                
            for block_part in block_part_range:
                from_idx = block_part*(BLOCKS_CONTEXT)
                to_idx = (block_part+1)*(BLOCKS_CONTEXT)
                
                x1 = s[:, from_idx:to_idx, :].to(device) # torch.Size([128, 30, 3, 64, 112])
                y1 = a[:, from_idx:to_idx, :].to(device).float()
                r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(device).float() 
                t1 = timesteps[:, from_idx:to_idx].to(device)
                masks1 = masks[:, from_idx:to_idx].to(device)
                    
                # print('2', x1.shape)
                model.flag = 1 if block_part == max(block_part_range) else 0
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
                    if config["training_config"]["grad_norm_clip"] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_config"]["grad_norm_clip"])
                    optimizer.step()
                    scheduler.step()
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    if wwandb:
                        wandb.log({"learning_rate": lr})
            it_counter += 1 
            
        #if model.flag == 1:
        pbar.set_description(f"ep {epoch+1} it {it} tTotal {train_loss.item():.2f} lr {lr:e}")     
        
        # Scheduler changer
        if it_counter >= config["training_config"]["warmup_steps"] and switch == False:
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=learning_rate_decay, patience=patience, mode="min")
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                          start_factor=1.0, 
                                                          end_factor=0.01,
                                                          total_iters=max_segments*len(train_dataloader)*config["training_config"]["epochs"])
            switch = True
        
        # Save
        if (epoch + 1) % 100 == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 50 or (epoch + 1) == 25 or (epoch + 1) == 1:
            if config["training_config"]["online_inference"] == True:
                if config["model_config"]["mode"] == 'doom':
                    model.eval()
                    for ret in [config["online_inference_config"]["desired_return"]]:
                        returns = []
                        ts = []
                        for i in range(len(seeds_list)):
                            episode_return, act_list, t, _, _ = get_returns_VizDoom(model=model, ret=ret, seed=seeds_list[i], 
                                                                                    episode_timeout=episode_timeout, 
                                                                                    context_length=config["training_config"]["context_length"], 
                                                                                    device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                    config=config,
                                                                                    mean=MEAN,
                                                                                    std=STD,
                                                                                    use_argmax=use_argmax, create_video=False)

                            returns.append(episode_return)
                            ts.append(t)
                            pbar.set_description(f"Online inference: [{i+1} / {len(seeds_list)}] Time: {t}, Return: {episode_return:.2f}")

                        if wwandb:
                            wandb.log({"LifeTime":  np.mean(ts), "return": np.mean(returns)})
                elif config["model_config"]["mode"] == 'memory_maze':
                    model.eval()
                    for ret in [config["online_inference_config"]["desired_return"]]:
                        returns = []
                        ts = []
                        for i in range(len(seeds_list)):
                            episode_return, act_list, t, _, _ = get_returns_MemoryMaze(model=model, ret=ret, seed=seeds_list[i], 
                                                                                    episode_timeout=episode_timeout, 
                                                                                    context_length=config["training_config"]["context_length"], 
                                                                                    device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                    config=config,
                                                                                    mean=MEAN,
                                                                                    std=STD,
                                                                                    use_argmax=use_argmax, create_video=False)

                            returns.append(episode_return)
                            ts.append(t)
                            pbar.set_description(f"Online inference: [{i+1} / {len(seeds_list)}] Time: {t}, Return: {episode_return:.2f}")

                        if wwandb:
                            wandb.log({"LifeTime":  np.mean(ts), "return": np.mean(returns)})

            
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": wandb_step})
            # torch.save(model.state_dict(), ckpt_path + '_save' + '_KTD.pth')
            torch.save(model.state_dict(), ckpt_path + str(epoch+1) + '_KTD.pth')
            
    return model