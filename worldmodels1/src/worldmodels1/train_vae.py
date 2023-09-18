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
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
torch.cuda.set_device(rank)

# Argument parser setup
logging.info("Parsing arguments")
parser = argparse.ArgumentParser(description='Train VAE model for Car Racing')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
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
        x = x.unsqueeze(0)  # Add channel dimension 
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

#Dictonary to store the loss metrics
losses = {'total_loss': [], 'recon_loss': [], 'kl_div': []}

# Training loop
logging.info("Starting training loop")
best_loss = float('inf')
for epoch in range(args.epochs):
    # Wrap dataloader with tqdm to create a progress bar
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
        for batch_idx, batch in enumerate(dataloader):
            states = batch.to(device)
            
            # Forward pass
            recon_states, mu, logvar = vae(states)
            
            # Loss computation
            recon_loss = reconstruction_loss(recon_states, states)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + args.beta * kl_div
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'Total Loss': loss.item(), 'Recon Loss': recon_loss.item(), 'KL Div': kl_div.item()})
            pbar.update(1)
        
    logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Total Loss: {loss.item()}, Reconstruction Loss: {recon_loss.item()}, KL Divergence: {kl_div.item()}')
    #track loss metrics
    losses['total_loss'].append(loss.item())
    losses['recon_loss'].append(recon_loss.item())
    losses['kl_div'].append(kl_div.item())

    #track best loss  
    if loss.item() < best_loss:
        best_loss = loss.item()
        #save model 
        torch.save(vae.state_dict(), 'vae2.pth')
        logging.info('Model saved')
        #save loss metrics to json file
        with open('losses.json', 'w') as f:
            json.dump(losses, f)
        logging.info('Losses saved')



