import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cnnvae import VAE

# Argument parser setup
parser = argparse.ArgumentParser(description='Train VAE model for Car Racing')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data')
parser.add_argument('--beta', type=float, default=1.0, help='Weight for KL Divergence term')
args = parser.parse_args()

# Define your VAE, Dataset, and DataLoader classes as before
class CarRacingDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

dataset = CarRacingDataset(preprocessed_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


# Initialize the VAE model and optimizer
vae = VAE()
if torch.cuda.device_count() > 1:
    vae = nn.DataParallel(vae)
vae = vae.to(device="cuda")

optimizer = optim.Adam(vae.parameters(), lr=args.lr)

# Load your preprocessed data
# Assume the data is loaded into a variable called preprocessed_data
# preprocessed_data = load_data(args.data_path)

dataset = CarRacingDataset(preprocessed_data)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Loss criterion
reconstruction_loss = nn.MSELoss(reduction='sum')

# Training loop
for epoch in range(args.epochs):
    for batch_idx, batch in enumerate(dataloader):
        states = torch.stack([torch.tensor(t[0], dtype=torch.float32) for t in batch]).to(device="cuda")
        
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

    print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item()}')
