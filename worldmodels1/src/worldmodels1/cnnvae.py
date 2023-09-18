import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # Output size: 32x32x32
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output size: 16x16x64
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output size: 8x8x128
        self.enc_fc1 = nn.Linear(8 * 8 * 128, 256)
        self.enc_fc2_mean = nn.Linear(256, 32)
        self.enc_fc2_logvar = nn.Linear(256, 32)

        # Decoder
        self.dec_fc1 = nn.Linear(32, 256)
        self.dec_fc2 = nn.Linear(256, 8 * 8 * 128)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # Output size: 16x16x64
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # Output size: 32x32x32
        self.dec_conv1 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # Output size: 64x64x1

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
        z = z.view(z.size(0), 128, 8, 8)
        z = torch.relu(self.dec_conv3(z))
        z = torch.relu(self.dec_conv2(z))
        z = torch.sigmoid(self.dec_conv1(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
