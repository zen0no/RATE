import torch
import wandb
from tqdm import tqdm
import numpy as np

# from RATE_GTrXL import mem_transformer_v2_GTrXL
# from RATE_GTrXL import mem_transformer_v2_GTrXL_ca
from RATE_GTrXL import mem_transformer_v2_GTrXL_gru

from TMaze_new.TMaze_new_src.inference.val_tmaze import get_returns_TMaze
from TMaze_new.TMaze_new_src.utils.additional2 import plot_cringe 
from TMaze_new.TMaze_new_src.utils import seeds_list
from TMaze_new.TMaze_new_src.train import FactorScheduler

def train(model, optimizer, scheduler, raw_model, new_segment, epochs_counter, segments_count, wandb_step, ckpt_path, config, train_dataloader, val_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the config dictionary to initialize the model
    if model is None:
        model = mem_transformer_v2_GTrXL_gru.MemTransformerLM(**config["model_config"])

        model.loss_last_coef = config["training_config"]["coef"]
        torch.nn.init.xavier_uniform_(model.r_w_bias);
        torch.nn.init.xavier_uniform_(model.r_r_bias);
        wandb_step  = 0
        epochs_counter = 0
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training_config"]["learning_rate"], 
                                      weight_decay=config["training_config"]["weight_decay"], 
                                      betas=(config["training_config"]["beta_1"], config["training_config"]["beta_2"]))
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/config["warmup_steps"], 1))
        scheduler = FactorScheduler(optimizer, factor=1.0, stop_factor_lr=config["training_config"]["lr_end_factor"], 
                                    base_lr=config["training_config"]["learning_rate"], total_iterations = config["training_config"]["epochs"] * config["training_config"]["max_segments"],
                                    max_segments = config["training_config"]['max_segments'], warmup_steps=config["training_config"]["warmup_steps"], max_epochs=config["training_config"]["epochs"])
        raw_model = model.module if hasattr(model, "module") else model

        # !!!!!!!!!!!!!!!!!!!!!!
        print(config)
        # !!!!!!!!!!!!!!!!!!!!!!
    
    if new_segment == True: # !
        scheduler.current_iteration = 0
        scheduler.new_segment = True
        scheduler.warmup_counter = 0
        scheduler.segments_count = segments_count
        
        new_segment = False
        
    model.to(device)
    model.train()
    
    wwandb = config["wandb_config"]["wwandb"]
    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    best_val_loss = np.inf
    epochs_without_improvement = 0
    max_epochs_without_improvement = 50

    EFFECTIVE_SIZE_BLOCKS = config["training_config"]["context_length"] * config["training_config"]["sections"]
    BLOCKS_CONTEXT = config["training_config"]["context_length"]

    scheduler.warmup_steps *= EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT # !
    scheduler.total_iterations *= EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT # !

    switch = False
    
    pbar = tqdm(range(config["training_config"]["epochs"]))
    for epoch in pbar:
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch
            #print('1', s.shape)
            memory = None
            mem_tokens = None
            
            block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
                
            for block_part in block_part_range:
                from_idx = block_part*(BLOCKS_CONTEXT)
                to_idx = (block_part+1)*(BLOCKS_CONTEXT)

                x1 = s[:, from_idx:to_idx, :].to(device)
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
                    # print('before', memory)
                    res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                    memory = res[0][2:]
                    # print("a", len(memory), memory[0].shape, memory[0][0, 0, 0:5])
                    logits, loss = res[0][0], res[0][1]
                    mem_tokens = res[1]
        
                    train_loss_all = model.loss_all
                    if model.flag == 1:
                        train_loss_last = model.loss_last

                    if wwandb and model.flag == 1:
                        wandb.log({"train_last_loss": train_loss_last.item(), 
                                   "train_loss": train_loss_all.item(), 
                                   "train_accuracy": model.accuracy, 
                                   "train_last_acc": model.last_acc})

                if is_train:
                    model.zero_grad()
                    optimizer.zero_grad()
                    train_loss_all.backward(retain_graph=True)
                    if config["training_config"]["grad_norm_clip"] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_config"]["grad_norm_clip"])
                    optimizer.step()
                    # ! 
                    
                    # !
                    scheduler.step()
                    #scheduler.step()
                    scheduler.new_segment = False # !
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
            it_counter += 1 
            epochs_counter += 1
            #scheduler.current_iteration = epoch
        
        # Val
        model.eval()
        is_train = False
        with torch.no_grad():
            for it, batch in enumerate(val_dataloader):        
                s, a, rtg, d, timesteps, masks = batch
                memory = None
                mem_tokens = None
                #print('3', s.shape)
                block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
                    
                for block_part in block_part_range:
                    from_idx = block_part*(BLOCKS_CONTEXT)
                    to_idx = (block_part+1)*(BLOCKS_CONTEXT)
                    x1 = s[:, from_idx:to_idx, :].to(device)
                    y1 = a[:, from_idx:to_idx, :].to(device).float()
                    r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(device).float() 
                    t1 = timesteps[:, from_idx:to_idx].to(device)
                    masks1 = masks[:, from_idx:to_idx].to(device)
                        
                    #print('4', x1.shape)
                    model.flag = 1 if block_part == max(block_part_range) else 0
                    if mem_tokens is not None:
                        mem_tokens = mem_tokens.detach()
                    elif raw_model.mem_tokens is not None:
                        mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)
                    with torch.set_grad_enabled(is_train):
                        optimizer.zero_grad()
                        res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                        memory = res[0][2:]
                        logits, loss = res[0][0], res[0][1]
                        mem_tokens = res[1]
                        
                        val_loss_all = model.loss_all
                        if model.flag == 1:
                            val_loss_last = model.loss_last

                        if wwandb and model.flag == 1:
                            wandb.log({"val_last_loss": val_loss_last.item(), 
                                       "val_loss": val_loss_all.item(), 
                                       "val_accuracy": model.accuracy, 
                                       "val_last_acc": model.last_acc,
                                       "learning_rate": lr})
                    if model.flag == 1:
                        pbar.set_description(f"ep {epoch+1} it {it} tTotal {train_loss_all.item():.2f} vTotal {val_loss_all.item():.2f} lr {lr:e}")

        # Early stopping
        if val_loss_all < best_val_loss:
            best_val_loss = val_loss_all
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Проверка условия early stopping
        if epochs_without_improvement >= max_epochs_without_improvement and config["training_config"]["use_erl_stop"] == True:
            print("Early stopping!")
            break        

        # # Scheduler changer
        # if it_counter >= config["training_config"]["warmup_steps"] and switch == False: # !
        #     #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=learning_rate_decay, patience=patience, mode="min")
        #     scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=config["training_config"]["epochs"]*config["training_config"]["max_segments"])
        #     switch = True
        
        if wwandb:
            wandb.log({"segments_count": segments_count})
        
        # Save 
        if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or epoch == 0:
            if config["training_config"]["online_inference"]:
                model.eval()
                with torch.no_grad():
                    goods, bads = 0, 0
                    timers = []
                    rewards = []
                    seeds = seeds_list
                    pbar2 = range(len(seeds))
                    for indx, iii in enumerate(pbar2):
                        # episode_return, act_list, t, _ , delta_t, attn_map = get_returns_TMaze(model=model, ret=1.0, seed=seeds[iii], episode_timeout=config["episode_timeout"],
                        #                                                                         corridor_length=config["corridor_length"], context_length=config["context_length"],
                        #                                                                         device=device, act_dim=config["act_dim"], config=config, create_video=False)
                        episode_return, act_list, t, _ , delta_t, attn_map = get_returns_TMaze(model=model, ret=config["data_config"]["desired_reward"], 
                                                                                               seed=seeds[iii], 
                                                                                               episode_timeout=config["online_inference_config"]["episode_timeout"],
                                                                                               corridor_length=config["online_inference_config"]["corridor_length"], 
                                                                                               context_length=config["training_config"]["context_length"],
                                                                                               device=device, act_dim=config["model_config"]["ACTION_DIM"],
                                                                                               config=config, create_video=False)
                        if episode_return == config["data_config"]["desired_reward"]:
                            goods += 1
                        else:
                            bads += 1
                        timers.append(delta_t)
                        rewards.append(episode_return)
                        
                        # if seeds[iii] == 2:
                        #     C = plot_cringe(attn_map, config["online_inference_config"]["corridor_length"], config, config["model_config"]["mem_at_end"])
                        #     if wwandb:
                        #         Cs = wandb.Image(C, caption=f"Val attention map for the seed {seeds[indx]} with L = {config["online_inference_config"]['corridor_length']} and T = {config["online_inference_config"]['episode_timeout']}")
                        #         wandb.log({"inference_attention_maps": Cs})
                            
                    suc_rate = goods / (goods + bads)
                    ep_time = np.mean(timers)

                    # !!!!!!!!!!!!
                    #print("Succes Rate:", suc_rate)
                    # !!!!!!!!!!!!

                    if wwandb:
                        wandb.log({"Success_rate": suc_rate, "Mean_D[time]": ep_time})
        
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": wandb_step})
            #torch.save(model.state_dict(), ckpt_path + '_' + str(wandb_step) + '_KTD.pth')
            torch.save(model.state_dict(), ckpt_path + '_save' + '_KTD.pth')

    scheduler.warmup_steps /= EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT # !
    scheduler.total_iterations /= EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT # !
    return model, wandb_step, optimizer, scheduler, raw_model, epochs_counter