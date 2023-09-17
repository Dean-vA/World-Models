import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from cnnvae import VAE
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

dist.init_process_group(backend='nccl')

# Argument parser setup
logging.info("Parsing arguments")
parser = argparse.ArgumentParser(description='Train VAE model for Car Racing')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data')
parser.add_argument('--beta', type=float, default=1.0, help='Weight for KL Divergence term')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
args = parser.parse_args()
logging.info(f'Arguments parsed: {args}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')
#number of gpus
logging.info(f'Number of gpus: {torch.cuda.device_count()}')

# Load your saved data
logging.info(f'Loading data from {args.data_path}')
preprocessed_data = np.load(args.data_path, allow_pickle=True)
logging.info('Data loaded successfully')

class CarRacingDataset(Dataset):
    def __init__(self, preprocessed_data):
        # Only take the 'state' part of each tuple (i.e., the first element)
        self.data = [episode[i][0] for episode in preprocessed_data for i in range(len(episode))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.from_numpy(x).float() / 255.0  # Convert to float and normalize
        #unsqueeze to add a dimension of size one at the specified position
        x = x.unsqueeze(1)  # Add channel dimension [batch, channel, height, width]
        return x

dataset = CarRacingDataset(preprocessed_data)


if torch.cuda.device_count() > 1:
    # Initialize the DistributedSampler
    sampler = DistributedSampler(dataset)
    # Create DataLoader with the sampler
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=False,  # set to False
                            num_workers=args.num_workers, 
                            pin_memory=True,  # Optional but can improve performance with GPU
                            sampler=sampler)  # use the DistributedSampler
else:
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Initialize the VAE model and optimizer
logging.info("Initializing VAE model and optimizer")
vae = VAE().to(device)
if torch.cuda.device_count() > 1:
    vae = DistributedDataParallel(vae)
logging.info(f'Number of GPUs used: {torch.cuda.device_count()}')

optimizer = optim.Adam(vae.parameters(), lr=args.lr)
logging.info("Optimizer initialized")

# Loss criterion
reconstruction_loss = nn.MSELoss(reduction='sum')

# Training loop
logging.info("Starting training loop")
for epoch in range(args.epochs):
    for batch_idx, batch in enumerate(dataloader):
        logging.info(f'Starting batch {batch_idx}/{len(dataloader)}')
        states = batch.to(device)
        
        # Forward pass
        print("Shape of states:", states.shape)
        recon_states, mu, logvar = vae(states)
        
        # Loss computation
        recon_loss = reconstruction_loss(recon_states, states)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + args.beta * kl_div
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.dinfo(f'Completed batch {batch_idx} with loss {loss.item()}')

    logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item()}')

