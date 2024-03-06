import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class MemoryMazeDataset(Dataset):
    def __init__(self, directory, gamma, max_length, only_non_zero_rewards):
        """_summary_

        Args:
            directory (str): path to the directory with data files
            gamma (float): discount factor
            max_length (int): maximum number of timesteps used in batch generation
                                (max in dataset: 1001)
            only_non_zero_rewards (bool): if True then use only trajectories
                                            with non-zero reward in the first
                                            max_length timesteps
        """
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.gamma = gamma
        self.max_length = max_length
        self.filtered_list = []
        self.only_non_zero_rewards = only_non_zero_rewards
        if self.only_non_zero_rewards:
            print('Filtering data...')
            self.filter_trajectories()

    def discount_cumsum(self, x):
        """
        Compute the discount cumulative sum of a 1D array.

        Args:
            x (ndarray): 1D array of values.

        Returns:
            ndarray: Discount cumulative sum of the input array.
        """
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum
    
    def filter_trajectories(self):
        for idx in tqdm(range(len(self.file_list)), total=len(self.file_list)):
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = np.load(file_path)
            if any(data['reward'][:self.max_length]) > 0:
                self.filtered_list.append(self.file_list[idx])

    def __len__(self):
        if self.only_non_zero_rewards:
            return len(self.filtered_list)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.only_non_zero_rewards:
            file_path = os.path.join(self.directory, self.filtered_list[idx])
        else:
            file_path = os.path.join(self.directory, self.file_list[idx])
        data = np.load(file_path)

        image = torch.from_numpy(data['image']).permute(0, 3, 1, 2).float()
        action = torch.from_numpy(data['action'])
        # reward = torch.from_numpy(data['reward']).unsqueeze(-1)
        rtg = torch.from_numpy(self.discount_cumsum(data['reward'])).unsqueeze(-1)
        timesteps = torch.from_numpy(np.arange(0, self.max_length).reshape(-1))
        done = torch.zeros_like(timesteps)
        mask = torch.ones_like(timesteps)

        # print(image.shape, action.shape, rtg.shape, timesteps.shape, mask.shape)
        
        image = image[:self.max_length, :, :, :]
        action = action[:self.max_length, :]
        rtg = rtg[:self.max_length, :]

        return image, action, rtg, done, timesteps, mask

# Assuming 'directory_path' is the path to the directory containing .npz files
# dataset = MemoryMazeDataset('MemoryMaze/MemoryMaze_data/9x9/', gamma=1.0, max_length=1001, only_non_zero_rewards=True)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
