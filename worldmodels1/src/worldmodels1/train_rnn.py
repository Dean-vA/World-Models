import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from utils import get_dataloader #TO DO THIS NEEDS TO BE A DIFFERENT DATALOADER FOR THE SEQUENCE DATA
from mdnrnn import MemoryModel  # Import MemoryModel from mdnrnn.py


# Moved the argparse section to here so that the arguments can be parsed before the 'if __name__ == '__main__':'
parser = argparse.ArgumentParser(description='Train Memory Model')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in LSTM')
parser.add_argument('--n_gaussians', type=int, default=5, help='Number of Gaussians in MDN')
parser.add_argument('--data_path', type=str, help='Path to the training data')
parser.add_argument('--save_path', type=str, default='./memory_model.pth', help='Where to save the model')
parser.add_argument('--seq_len', type=int, default=50, help='Sequence length for LSTM input')  # Added seq_len argument
args = parser.parse_args()


# Define loss function
def loss_function(pi, mu, sigma, y):
    result = Normal(loc=mu, scale=sigma).log_prob(y)
    result += torch.log(pi)
    result = torch.logsumexp(result, dim=1)
    return -torch.mean(result)


# Main training loop
if __name__ == '__main__':
    # Initialize the model and optimizer
    model = MemoryModel(n_input=32, n_hidden=args.n_hidden, n_output=32, n_gaussians=args.n_gaussians)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load your dataset
    dataset = get_dataloader(args.data_path, seq_len=args.seq_len)  # Implement your DataLoader to handle sequence length
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            pi, mu, sigma = model(batch['input'])  # Your DataLoader should yield a dict with 'input' and 'target'

            # Compute loss
            loss = loss_function(pi, mu, sigma, batch['target'])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Logging
            if i % 10 == 0:  # Log every 10 batches
                print(f'Epoch [{epoch + 1}/{args.epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), args.save_path)
