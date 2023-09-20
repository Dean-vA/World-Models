import torch
import torch.nn as nn
import torch.optim as optim

class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(n_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_gaussians * 3)
        )
    def forward(self, x):
        print(f'x shape: {x.shape}')
        z_h = self.z_h(x)
        print(f'z_h shape: {z_h.shape}')
        pi, mu, sigma = torch.split(z_h, 3, dim=1)
        pi = nn.Softmax(dim=1)(pi)
        sigma = torch.exp(sigma)
        return pi, mu, sigma

class MemoryModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_gaussians):
        super(MemoryModel, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True)
        self.mdn = MDN(n_hidden, n_gaussians)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pi, mu, sigma = self.mdn(lstm_out)
        return pi, mu, sigma

