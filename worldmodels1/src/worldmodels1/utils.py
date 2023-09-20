import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CarRacingDataset(Dataset):
    def __init__(self, preprocessed_data, get_action=False, get_metadata=False):
        self.get_action = get_action
        logging.info(f'get_action: {get_action}')
        self.get_meta = get_metadata
        logging.info(f'get_metadata: {get_metadata}')
        self.imgdata = [episode[i][0] for episode in preprocessed_data for i in range(len(episode))]
        self.actiondata = [episode[i][1] for episode in preprocessed_data for i in range(len(episode))]
        # Metadata is episode number and step count
        self.episodedata = [episode[i][4] for episode in preprocessed_data for i in range(len(episode))]
        self.step_countdata = [episode[i][5] for episode in preprocessed_data for i in range(len(episode))]

    def __len__(self):
        return len(self.imgdata)

    def __getitem__(self, index):
        x = self.imgdata[index]
        #logging.info(f'Getting item at index {index}, with action {self.actiondata[index]}')
        x = torch.from_numpy(x).float() / 255.0
        x = x.unsqueeze(0)
        if self.get_action:
            if self.get_meta:
                return x, self.actiondata[index], self.episodedata[index], self.step_countdata[index]
            return x, self.actiondata[index]
        return x

def get_dataloader(preprocessed_data, batch_size, num_workers, get_action=False, shuffle=True, get_metadata=False):
    logging.info(f'get_metadata: {get_metadata}')
    dataset = CarRacingDataset(preprocessed_data, get_action=get_action, get_metadata=get_metadata)
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)
    else:
        logging.info(f'Shuffle: {shuffle}')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class MemoryModelDataset(Dataset):
    def __init__(self, latent_action_pairs, seq_length=999, latent_dim=32):
        self.latent_action_pairs = latent_action_pairs
        self.seq_length = seq_length
        self.latent_dim = latent_dim # Make sure this matches with `n_input` in MemoryModel

    def __len__(self):
        return len(self.latent_action_pairs) - self.seq_length  # To avoid out-of-index errors

    def __getitem__(self, index):
        input_sequence = self.latent_action_pairs[index:index+self.seq_length]
        # teacher forcing (shift by one predictions) only inlcude the latent vectors
        target = self.latent_action_pairs[index+1:index+self.seq_length+1][:, :self.latent_dim]
        return torch.FloatTensor(input_sequence), torch.FloatTensor(target)
