import sys
import os
os.environ["LD_LIBRARY_PATH"]="/.mujoco/mujoco210/bin"
os.environ["LD_LIBRARY_PATH"]="/home/jovyan/.mujoco/mujoco210/bin"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WANDB_MODE'] = 'online'


import gym
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import mem_transformer
from tqdm import tqdm

import argparse
import pickle
import random
import sys
import os
import glob
from colabgymrender.recorder import Recorder
sys.path.append('rmdt_transformer/gym/')
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg


#pip install imageio-ffmpeg


from utils.eval_functions import get_batch, get_returns

ENVS = [['halfcheetah-medium-v2','checkpoints_Cheetah_Med',17,6,1000,6000],
        ['halfcheetah-medium-replay-v2','checkpoints_Cheetah_Med_Repl',17,6,1000,6000],
        ['halfcheetah-expert-v2','checkpoints_Cheetah_Expert',17,6,1000,6000],
       
        ['walker2d-medium-v2','checkpoints_Walker_Med',17,6,1000,5000],
        ['walker2d-medium-replay-v2','checkpoints_Walker_Med_Repl',17,6,1000,5000],
        ['walker2d-expert-v2','checkpoints_Walker_Expert',17,6,1000,5000],
        
        ['hopper-medium-v2','checkpoints_Hopper_Med',11,3,1000,3600],
        ['hopper-medium-replay-v2','checkpoints_Hopper_Med_Repl',11,3,1000,3600],
        ['hopper-expert-v2','checkpoints_Hopper_Expert',11,3,1000,3600],
       ]

INIT_ENV = False

class Args:
    def __init__(self):
        self.seed=123
        #self.context_length=user_config.context_length
        #self.epochs=user_config.epochs
        self.num_steps= 500000
        self.num_buffers=50
        #self.game=user_config.game
        self.batch_size=128
        self.nemb=128
        self.data_dir_prefix='../data/'
        self.trajectories_per_buffer=10
        self.use_scheduler=True
        #self.ckpt_path = user_config.save_path

        self.vocab_size = 100
        self.n_layer = 3
        self.n_head = 1
        self.d_model = 128
        self.d_head = 128
        self.d_inner = 128
        self.dropout = 0.1
        self.dropatt = 0.05
        self.MEM_LEN = 2               
        self.ext_len = 1
        self.tie_weight = False
        self.num_mem_tokens = 3*5      
        self.mem_at_end = True         
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.betas = (0.9, 0.95)
        self.warmup_steps = 100
        self.grad_norm_clip = 1.0
        self.max_timestep = 10000
        self.context_length = 20
        self.sections = 3
        self.num_spets_per_epoch = 1000
        self.is_train = True
        self.wwandb = False
        
        self.batch_size = 2048*2
        self.num_eval_episodes = 10
        self.pct_traj = 1.
        self.max_ep_len = 10000
        self.scale = 1000.


class Agent:
    def __init__(self,args):
        self.args = args
        self.device = torch.device('cuda:0')

    def load_dataset(self):
        args = self.args

        gym_name = ENVS[args.env_id]
        
        game_gym_name = gym_name[0]
        env_name = game_gym_name.split('-')[0]
        dataset = '-'.join(game_gym_name.split('-')[1:-1])
        state_dim = gym_name[2]
        act_dim = gym_name[3]
        self.max_ep_len = gym_name[4]
        ret_global = gym_name[5]
        use_recorder = False
        
        if INIT_ENV:
            self.env = gym.make(game_gym_name)
            directory = '../video'
            if use_recorder:
                self.env = Recorder(self.env, directory, fps=30)
    
        # load dataset
        dataset_path = f'../data/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        states, traj_lens, returns = [], [], []
        for path in self.trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)  
        num_timesteps = sum(traj_lens)
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        self.state_mean_torch, self.state_std_torch = torch.from_numpy(self.state_mean).to(self.device), torch.from_numpy(self.state_std).to(self.device) 
        
        print('=' * 50)
        print(f'Starting new experiment: {env_name} {dataset}')
        print(f'{len(traj_lens)} self.trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)
    
        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = max(int(args.pct_traj*num_timesteps), 1)
        self.sorted_inds = np.argsort(returns)  # lowest to highest
        self.num_trajectories = 1
        timesteps = traj_lens[self.sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[self.sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[self.sorted_inds[ind]]
            self.num_trajectories += 1
            ind -= 1
        self.sorted_inds = self.sorted_inds[-self.num_trajectories:]
        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = traj_lens[self.sorted_inds] / sum(traj_lens[self.sorted_inds])
        
        s, a, r, d, rtg, timesteps, mask = get_batch(self.trajectories,self.state_mean,self.state_std,self.num_trajectories,self.max_ep_len,args.state_dim,args.act_dim,self.sorted_inds,self.p_sample,self.device,batch_size=args.batch_size,max_len=args.EFFECTIVE_SIZE_BLOCKS)

    def load_model(self):
        args = self.args

        self.model = mem_transformer.MemTransformerLM(STATE_DIM=args.state_dim,ACTION_DIM=args.act_dim,n_token=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model,
        d_head=args.d_head, d_inner=args.d_inner, dropout=args.dropout, dropatt=args.dropatt, mem_len=args.MEM_LEN, ext_len=args.ext_len, tie_weight=args.tie_weight, num_mem_tokens=args.num_mem_tokens, mem_at_end=args.mem_at_end)
        torch.nn.init.xavier_uniform_(self.model.r_w_bias);
        torch.nn.init.xavier_uniform_(self.model.r_r_bias);
        self.model.train()
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate,  weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: min((steps+1)/args.warmup_steps, 1))
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model

    def train(self):
        args = self.args
        
        INTERVAL = 200
        losses = []
        wandb_step  = 0
        for epoch in range(args.max_epochs):
            pbar = tqdm(enumerate(list(range(args.num_spets_per_epoch))), total=args.num_spets_per_epoch)
            for it, i in pbar:
                s, a, r, d, rtg, timesteps, mask = get_batch(self.trajectories,self.state_mean,self.state_std,self.num_trajectories,self.max_ep_len,args.state_dim,args.act_dim,self.sorted_inds,self.p_sample,self.device,batch_size=args.batch_size,max_len=args.EFFECTIVE_SIZE_BLOCKS)
                s = s[torch.all(mask==1,dim=1)]
                a = a[torch.all(mask==1,dim=1)]
                rtg = rtg[torch.all(mask==1,dim=1)]
                timesteps = timesteps[torch.all(mask==1,dim=1)]

                memory = None
                mem_tokens=None

                for block_part in range(args.EFFECTIVE_SIZE_BLOCKS//args.BLOCKS_CONTEXT):

                    from_idx = block_part*(args.BLOCKS_CONTEXT)
                    to_idx = (block_part+1)*(args.BLOCKS_CONTEXT)
                    x1 = s[:, from_idx:to_idx, :].to(self.device)
                    y1 = a[:, from_idx:to_idx, :].to(self.device)
                    r1 = rtg[:,:-1,:][:, from_idx:to_idx, :].to(self.device) #### SWITCH TO RTG
                    t1 = timesteps[:, from_idx:to_idx].to(self.device)

                    if mem_tokens is not None:
                        mem_tokens = mem_tokens.detach()
                    elif self.raw_model.mem_tokens is not None:
                        mem_tokens = self.raw_model.mem_tokens.repeat(1, x1.shape[0], 1)

                    with torch.set_grad_enabled(args.is_train):
                        if memory is not None:
                            res = self.model(x1, y1, r1, y1,t1, *memory, mem_tokens=mem_tokens)
                        else:
                            res = self.model(x1, y1, r1, y1,t1, mem_tokens=mem_tokens)
                        memory = res[0][2:]
                        logits, loss = res[0][0], res[0][1]
                        losses.append(loss.item())
                        #print('Loss: ',loss.item())
                        if args.wandb:
                            wandb.log({"train_loss":  loss.item()})
                            
                    if args.is_train:
                        # backprop and update the parameters
                        self.model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_norm_clip)
                        self.optimizer.step()
                        self.scheduler.step()
                        # decay the learning rate based on our progress
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")     
                
                if it % INTERVAL == 0:
                    
                    if INIT_ENV:
                        rews = []
                        steps = 10
                        prompt_steps = 1
                        pbar = tqdm(enumerate(list(range(steps))), total=steps)
                        for it, i in pbar:
                            eval_return, self.env = get_returns(self.model, self.env,args.ret_global, args.context_length, args.state_dim, args.act_dim, self.state_mean_torch, self.state_std_torch, self.max_ep_len, self.device, prompt_steps=prompt_steps, memory_20step=True, without_memory=False)
                            rews.append(eval_return)

                        print(f"Prompt_steps: {prompt_steps}  Mean Reward: {np.mean(np.array(rews)):.5f}. STD Reward: {np.std(np.array(rews)):.5f}")        
                        if args.wandb:            
                            wandb.log({"episode_STD":  np.std(rews)})#, step=wandb_step)
                            wandb.log({"episode_MEAN":  np.mean(rews)})#, step=wandb_step)

                        rews = []
                        steps = 10
                        prompt_steps = args.context_length
                        pbar = tqdm(enumerate(list(range(steps))), total=steps)
                        for it, i in pbar:
                            eval_return, self.env = get_returns(self.model, self.env,args.ret_global, args.context_length, args.state_dim, args.act_dim, self.state_mean_torch, self.state_std_torch, self.max_ep_len, self.device, prompt_steps=prompt_steps, memory_20step=True, without_memory=False)
                            rews.append(eval_return)

                        print(f"Prompt_steps: {prompt_steps}  Mean Reward: {np.mean(np.array(rews)):.5f}. STD Reward: {np.std(np.array(rews)):.5f}")        
                        if args.wandb:            
                            wandb.log({"{}_episode_STD".format(args.context_length):  np.std(rews)}) #, step=wandb_step)
                            wandb.log({"{}_episode_MEAN".format(args.context_length):  np.mean(rews)}) #, step=wandb_step)

                    wandb_step += 1     
                    torch.save(self.model.state_dict(), args.ckpt_path+'/'+str(wandb_step)+'.pth')
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("number_of_segments")
    parser.add_argument("segment_lenght")
    parser.add_argument("num_mem_tokens",default=15)
    
    args_INPUT = parser.parse_args()
    print(args_INPUT.env_id,args_INPUT.number_of_segments,args_INPUT.segment_lenght,args_INPUT.num_mem_tokens)


    #index = int(input('Choose env and dataset index: '))
    index = int(args_INPUT.env_id)
    
    gym_name = ENVS[index]

    args = Args()
    args.env_id = int(args_INPUT.env_id)
    args.context_length = int(args_INPUT.segment_lenght) #40
    args.sections = int(args_INPUT.number_of_segments) #3
    args.num_mem_tokens = int(args_INPUT.num_mem_tokens) #3*5

    
    args.block_size = 3*args.context_length
    args.EFFECTIVE_SIZE_BLOCKS = args.context_length * args.sections
    args.BLOCKS_CONTEXT = args.block_size//3
    args.init = 'uniform'
    args.init_range = 1
    args.init_std = 1
    args.game_gym_name = gym_name[0]
    args.env_name = args.game_gym_name.split('-')[0]
    args.dataset = '-'.join(args.game_gym_name.split('-')[1:-1])
    args.state_dim = gym_name[2]
    args.act_dim = gym_name[3]
    args.max_ep_len = gym_name[4]
    args.ret_global = gym_name[5]
    args.use_recorder = False
    args.ckpt_path = '../checkpoints_seed3/{}_ns_{}_sl_{}_nt_{}'.format('_'.join(args.game_gym_name.split('-')[:-1]),args_INPUT.number_of_segments,args_INPUT.segment_lenght,args_INPUT.num_mem_tokens)
    args.max_epochs = 20
    args.lr_decay = False
    args.wandb = True

    isExist = os.path.exists(args.ckpt_path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(args.ckpt_path)

    files = glob.glob(args.ckpt_path+'/*')
    for f in files:
        os.remove(f) 

    if args.wandb:
        idd = 'wandb_id' #
        wandb.init(project="your_project", name=args.ckpt_path.split('/')[-1], save_code=True, resume="allow")


    agent = Agent(args)

    agent.load_dataset()
    agent.load_model()
    agent.train()
