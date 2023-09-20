import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from utils import MemoryModelDataset
from mdnrnn import MemoryModel  # Import MemoryModel from mdnrnn.py
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Moved the argparse section to here so that the arguments can be parsed before the 'if __name__ == '__main__':'
parser = argparse.ArgumentParser(description='Train Memory Model')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in LSTM')
parser.add_argument('--n_gaussians', type=int, default=5, help='Number of Gaussians in MDN')
parser.add_argument('--data_path', type=str, help='Path to the training data')
parser.add_argument('--save_path', type=str, default='./memory_model.pth', help='Where to save the model')
parser.add_argument('--seq_len', type=int, default=999, help='Sequence length for LSTM input')  # Added seq_len argument
parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of VAE')  # Added latent_dim argument

args = parser.parse_args()

# Define loss function
def loss_function(pi, mu, sigma, y):
    result = Normal(loc=mu, scale=sigma).log_prob(y)
    result += torch.log(pi)
    result = torch.logsumexp(result, dim=1)
    return -torch.mean(result)


# Main training loop
if __name__ == '__main__':
    #check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Load your saved data
    latent_action_pairs = np.load(args.data_path, allow_pickle=True)
    logging.info('Data loaded successfully')

    # Initialize the model and optimizer
    model = MemoryModel(n_input=32, n_hidden=args.n_hidden, n_gaussians=args.n_gaussians).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logging.info("Model and optimizer initialized")

    # Load your dataset
    dataset = MemoryModelDataset(latent_action_pairs, seq_length=args.seq_len, latent_dim=args.latent_dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logging.info('Dataset and dataloader initialized')

    best_loss = np.inf
    for epoch in tqdm(range(args.epochs), desc='Epochs'):  # Wrap the epoch loop with tqdm
        batch_tqdm = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)  # Create a tqdm object for the dataloader loop
        for i, batch in enumerate(batch_tqdm):  
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            pi, mu, sigma = model(batch['input'].to(device))  # Ensure you use 'input' from the dict and send it to device

            # Compute loss
            loss = loss_function(pi, mu, sigma, batch['target'].to(device))  # Ensure you use 'target' from the dict and send it to device

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update tqdm
            if i % 10 == 0:  # Update every 10 batches
                batch_tqdm.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Save the model if it has the best loss
    if loss.item() < best_loss:
        torch.save(model.state_dict(), args.save_path)
        logging.info(f'Model saved to {args.save_path}, previous best loss: {best_loss:.4f}, current best loss: {loss.item():.4f}')