import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np

class CarRacingDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.data = [episode[i][0] for episode in preprocessed_data for i in range(len(episode))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.from_numpy(x).float() / 255.0
        x = x.unsqueeze(0)
        return x

def get_dataloader(preprocessed_data, batch_size, num_workers):
    dataset = CarRacingDataset(preprocessed_data)
    if torch.cuda.device_count() > 1:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

class MemoryModelDataset(Dataset):
    def __init__(self, latent_action_pairs, seq_length=1000, latent_dim=32):
        self.latent_action_pairs = latent_action_pairs
        self.seq_length = seq_length

    def __len__(self):
        return len(self.latent_action_pairs) - self.seq_length  # To avoid out-of-index errors

    def __getitem__(self, index):
        input_sequence = self.latent_action_pairs[index:index+self.seq_length]
        target = self.latent_action_pairs[index+self.seq_length][:self.latent_dim]  # Assuming the latent vector is the first 32 elements
        return {'input': input_sequence, 'target': target}
