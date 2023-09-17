import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.enc_fc1 = nn.Linear(128 * 6 * 6, 256)
        self.enc_fc2_mean = nn.Linear(256, 32)
        self.enc_fc2_logvar = nn.Linear(256, 32)

        # Decoder
        self.dec_fc1 = nn.Linear(32, 256)
        self.dec_fc2 = nn.Linear(256, 128 * 6 * 6)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)

    def encode(self, x):
        x = torch.relu(self.enc_conv1(x))
        x = torch.relu(self.enc_conv2(x))
        x = torch.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.enc_fc1(x))
        return self.enc_fc2_mean(x), self.enc_fc2_logvar(x)

    def decode(self, z):
        z = torch.relu(self.dec_fc1(z))
        z = torch.relu(self.dec_fc2(z))
        z = z.view(z.size(0), 128, 10, 10)
        z = torch.relu(self.dec_conv1(z))
        z = torch.relu(self.dec_conv2(z))
        z = torch.sigmoid(self.dec_conv3(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decode(z)
        return x_recon, mu, logvar
