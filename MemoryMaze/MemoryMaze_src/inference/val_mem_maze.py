import torch
import numpy as np
import gym
gym.logger.set_level(40)
import logging
logging.disable(logging.WARNING)
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

from TMaze_new.TMaze_new_src.utils import set_seed
from VizDoom.VizDoom_src.utils import z_normalize, inverse_z_normalize



@torch.no_grad()
def sample(model, x, block_size, steps, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        if saved_context is not None:
            results = model(x_cond, actions, rtgs,None, timestep, *saved_context, mem_tokens=mem_tokens)
        else:
            results = model(x_cond, actions, rtgs,None, timestep, mem_tokens=mem_tokens) 
            
        logits = results[0][0].detach()[:,-1,:]
        mem_tokens = results[1]
        memory = results[0][2:]
        
    return logits, mem_tokens, memory

def get_returns_MemoryMaze(model, ret, seed, episode_timeout, context_length, device, act_dim, config, mean, std, use_argmax=False, create_video=False):
    
    set_seed(seed)
    
    # * USE ONLY LAST MEM TOKEN:
    use_only_last_mem_token = False
    
    # * ADD NOISE TO MEM TOKENS:
    add_noise = False

    scale = 1
    max_ep_len = episode_timeout#* 3

    env = gym.make('memory_maze:MemoryMaze-9x9-v0', seed=seed)
    
    state = torch.tensor(env.reset()).float().permute(2, 0, 1) # model trained on [C, H, W], but env returns [H, W, C]
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
   
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True
    HISTORY_LEN = context_length
    
    rews = []
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    saved_context = None
    segment = 0
    prompt_steps = 0# 5
    
    act_list= []
    memories = []
    
    # !!!
    if use_only_last_mem_token:
        switcher = False
        saved_mem = None
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        if config["model_mode"] != 'DT' and config["model_mode"] != 'DTXL':
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[-1:,:]
                    states = states[:, -1:, :, :, :]
                    #target_return = target_return[:,-1:]
                    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
                    timesteps = timesteps[:, -1:]
                else:
                    actions = actions[-prompt_steps:,:]
                    states = states[:, -prompt_steps:, :, :, :]
                    target_return = target_return[:,-prompt_steps:]#+3600.
                    timesteps = timesteps[:, -prompt_steps:]
                    
                if t%(context_length)==0 and t>5:
                    # !!!
                    if use_only_last_mem_token:
                        mem_tokens = saved_mem
                    else:
                        mem_tokens = new_mem
                    
                    saved_context = new_notes
                    
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                    
        else:
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[1:,:]
                    states = states[:, 1:, :, :, :]
                    target_return = target_return[:,1:]
                    timesteps = timesteps[:, 1:]
                else:
                    actions = actions[-prompt_steps:,:]
                    states = states[:, -prompt_steps:, :, :, :]
                    target_return = target_return[:,-prompt_steps:]#+3600.
                    timesteps = timesteps[:, -prompt_steps:]
                    
                if t%(context_length)==0 and t>5:
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                    mem_tokens = new_mem
                    saved_context = new_notes

        #print(states.shape,actions.shape,target_return.shape,timesteps.shape)
        if t==0:
            act_to_pass = None
        else:
            act_to_pass = actions.unsqueeze(0)[:, 1:, :]
            if act_to_pass.shape[1] == 0:
                act_to_pass = None 
        
        #print(states.shape, target_return.shape, act_to_pass.shape if act_to_pass is not None else act_to_pass)
        # print(t, timesteps[:, -1])
        
        if add_noise:
            added_noise = torch.randn_like(mem_tokens)
        else:
            added_noise = torch.zeros_like(mem_tokens) if mem_tokens is not None else 0
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!
        b, l, c, h, w = states.shape
        states_norm = states.reshape(b*l, c, h, w)
        states_norm = z_normalize(states_norm, mean.to(device), std.to(device))
        states_norm = states_norm.reshape(b, l, c, h, w).to(device)

        # !!!!!!!!!!!!!!!!!!!
        try:
            sampled_action, new_mem, new_notes = sample(model=model,  
                                                        x=states_norm,
                                                        block_size=HISTORY_LEN, 
                                                        steps=1, 
                                                        sample=True, 
                                                        actions=act_to_pass, 
                                                        rtgs=target_return.unsqueeze(-1), 
                                                        timestep=timesteps, 
                                                        mem_tokens=mem_tokens,#torch.randn_like(mem_tokens),#mem_tokens+added_noise, 
                                                        saved_context=saved_context)
        except:
            print("ERROR!!!!!!!!!!!", t, states.shape, target_return.shape, act_to_pass.shape)
            
        # !!!!!!!
        if use_only_last_mem_token:
            if t > 0 and t % (context_length-1) == 0 and switcher == False:
                switcher = True
                saved_mem = new_mem
            
        if new_mem is not None:
            memories.append(mem_tokens.detach().cpu().numpy())
        #print(sampled_action[-1])
        
        # print(np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(), 3))
        if not use_argmax:
            act = np.random.choice(np.arange(env.action_space.n), p=torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        else:
            act = np.argmax(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        
        actions[-1] = act
        act_list.append(act)
        
        state, reward, done, info = env.step(act) # state [H, W, C], need [C, H, W]
        #print(t, reward)
        state = np.float32(state.transpose(2, 0, 1))
        state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
        
        out_states.append(state)
        
        rews.append(reward)
        #print(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[0,-1] - (reward/scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        # timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (1)], dim=1)
        episode_return += reward
        episode_length += 1
        
        if done:
            torch.cuda.empty_cache()
            break  
        
    if create_video == True:
        print("\n")
    
    env.close()
    return episode_return, act_list, t, out_states, memories