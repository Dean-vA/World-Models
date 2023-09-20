import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from utils import MemoryModelDataset
from mdnrnn import MemoryModel, mdn_loss  # Import MemoryModel from mdnrnn.py
import numpy as np
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # For headless servers
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Moved the argparse section to here so that the arguments can be parsed before the 'if __name__ == '__main__':'
parser = argparse.ArgumentParser(description='Train Memory Model')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in LSTM')
parser.add_argument('--n_gaussians', type=int, default=5, help='Number of Gaussians in MDN')
parser.add_argument('--data_path', type=str, help='Path to the training data')
parser.add_argument('--save_path', type=str, default='memory_model.pth', help='Where to save the model')
parser.add_argument('--seq_len', type=int, default=999, help='Sequence length for LSTM input')  # Added seq_len argument
parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of VAE')  # Added latent_dim argument
parser.add_argument('--action_dim', type=int, default=3, help='Action dimension')  # Added action_dim argument
args = parser.parse_args()
logging.info(f'Arguments parsed: {args}')

# Main training loop
if __name__ == '__main__':
    #check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Load your saved data
    latent_action_pairs = np.load(args.data_path, allow_pickle=True)
    logging.info('Data loaded successfully')

    # Initialize the model and optimizer
    model = MemoryModel(n_input=args.latent_dim+args.action_dim, n_hidden=args.n_hidden, n_gaussians=args.n_gaussians, latent_dim=args.latent_dim).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logging.info("Model and optimizer initialized")

    # Load your dataset
    dataset = MemoryModelDataset(latent_action_pairs, seq_length=args.seq_len, latent_dim=args.latent_dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logging.info('Dataset and dataloader initialized')

best_loss_epoch = np.inf
# Initialize empty list to hold loss values
loss_values = []
for epoch in tqdm(range(args.epochs), desc='Epochs'):  # Wrap the epoch loop with tqdm
    batch_tqdm = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)  # Create a tqdm object for the dataloader loop
    starting_loss = None  # To hold the starting loss value for each epoch
    for i, batch in enumerate(batch_tqdm):  
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        pi, mu, sigma = model(batch[0].to(device))  # [0] is the input sequence from the dataset
        if i == 0:
            logging.info(f'pi shape: {pi.shape}, mu shape: {mu.shape}, sigma shape: {sigma.shape}, y shape: {batch[1].shape}')
        
        # Compute loss
        loss = mdn_loss(batch[1].to(device), pi, mu, sigma)  # [1] is the target sequence from the dataset, loss here is averaged over the batch

        # Record the starting loss for this epoch, if it's the first iteration
        if starting_loss is None:
            starting_loss = loss.item()
        # Record the loss for this batch
        loss_values.append(loss.item())

        # Record the best loss so far, if it's lower than the previous best loss
        if loss.item() < best_loss:
            best_loss = loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update tqdm
        if i % 10 == 0:  # Update every 10 batches
            batch_tqdm.set_postfix({"Starting Loss": f"{starting_loss:.4f}", "Best Loss": f"{best_loss:.4f}", "Current Loss": f"{loss.item():.4f}"})  

        if i % 100 == 0: # Save the plot every 100 batches
            # Plotting the loss values
            plt.plot(loss_values)
            plt.xlabel('Batch Number')
            plt.ylabel('Loss')
            plt.title('Loss per Batch')
            plt.savefig('loss_per_batch.png')  # This saves the figure to the current working directory

    # Save the model if it has the best loss
    if loss.item() < best_loss_epoch:
        best_loss_epoch = loss.item()
        torch.save(model.state_dict(), args.save_path)
        logging.info(f'Model saved to {args.save_path}, previous best loss: {best_loss:.4f}, current best loss: {loss.item():.4f}')