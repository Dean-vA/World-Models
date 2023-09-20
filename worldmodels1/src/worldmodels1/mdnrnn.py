import torch
import torch.nn as nn
from torch.distributions import Normal

class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians, latent_dim):
        self.n_gaussians = n_gaussians
        self.latent_dim = latent_dim
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(n_hidden, 128),
            nn.ReLU(),
             nn.Linear(128, n_gaussians *( 2 * latent_dim+1))
        )
        self.first = True
    def forward(self, x):         
        z_h = self.z_h(x)                                                               ## TO DO FIGURE OUT IF PI should be [batch_size, seq_len, n_gaussians] or [batch_size, seq_len, n_gaussians*latent_dim]
        pi, mu, sigma = torch.split(z_h, [self.n_gaussians,                             # pi - mixture coefficients a single value for each gaussian mixture
                                          self.n_gaussians * self.latent_dim,           # mu - mean of each gaussian mixture for each latent dimension
                                          self.n_gaussians * self.latent_dim], dim=2)   # sigma - standard deviation of each gaussian mixture for each latent dimension

        pi = nn.Softmax(dim=2)(pi)
        sigma = torch.exp(sigma)
        if self.first:
            print(f'lstm output/mdn input shape: {x.shape}')
            print(f'z_h shape: {z_h.shape}')
            print(f'pi shape: {pi.shape}')
            print(f'mu shape: {mu.shape}')
            print(f'sigma shape: {sigma.shape}')
            self.first = False
        #reshape to match dimensions of the output
        pi = pi.view(pi.shape[0], pi.shape[1], self.n_gaussians)
        mu = mu.view(mu.shape[0], mu.shape[1], self.n_gaussians, self.latent_dim)
        sigma = sigma.view(sigma.shape[0], sigma.shape[1], self.n_gaussians, self.latent_dim)
        return pi, mu, sigma

class MemoryModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_gaussians, latent_dim):
        super(MemoryModel, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True)
        self.mdn = MDN(n_hidden, n_gaussians, latent_dim)
        self.first = True

    def forward(self, x):
        if self.first:
            print(f'input shape: {x.shape}')
            self.first = False
        lstm_out, _ = self.lstm(x)
        pi, mu, sigma = self.mdn(lstm_out)
        return pi, mu, sigma

def mdn_loss(y, pi, mu, sigma):
    # Expand dimensions of y to match the shape of the Gaussian parameters
    y = y.unsqueeze(2).expand_as(mu)
    
    # Create Gaussian distribution with the given mean and standard deviation
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    
    # Compute log probability of y under the Gaussian distribution
    log_prob = m.log_prob(y)
    
    # Make sure the dimensions of pi are consistent with log_prob
    pi = pi.unsqueeze(-1).expand_as(log_prob)  # adding the latent_dim axis THIS IS ONLY NEEDED IF PI IS [batch_size, seq_len, n_gaussians] not [batch_size, seq_len, n_gaussians*latent_dim]

    # Combine the log probabilities with the mixture weights
    log_prob_weighted = log_prob + torch.log(pi)
    
    # Compute the log sum using log-sum-exp trick for numerical stability
    max_log_prob_weighted = torch.max(log_prob_weighted, dim=2, keepdim=True)[0]
    log_prob_sum = max_log_prob_weighted + torch.log(torch.sum(torch.exp(log_prob_weighted - max_log_prob_weighted), dim=2, keepdim=True))
    
    # Return the negative log likelihood as the loss
    return -torch.mean(log_prob_sum)


