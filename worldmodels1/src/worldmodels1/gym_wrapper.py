# Awrapper for the car racing gym environment that preprocesses the input and returns the latent state of the vision model and hidden state of the memory model
import gymnasium as gym
from gymnasium import spaces

#add to path    
import sys
sys.path.append('worldmodels1/src/worldmodels1')

import numpy as np
from .cnnvae import VAE
from .mdnrnn import MemoryModel
import torch
from PIL import Image

class CarRacingWrapper(gym.Wrapper):
    def __init__(self, env, device='cpu'):
        super(CarRacingWrapper, self).__init__(env)
        # define observation space as vector of size 32 + 256
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32+256,), dtype=np.float32)

        self.vae = VAE()
        # Load the state_dict into CPU memory
        state_dict = torch.load('./../vae.pth', map_location='cpu')
        # Remove 'module.' prefix from state_dict keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Load the modified state_dict into the model
        self.vae.load_state_dict(new_state_dict)
        self.vae.eval()

        self.rnn = MemoryModel(n_input=32+3, n_hidden=256, n_gaussians=5, latent_dim=32)
        # load pretrained weights
        state_dict = torch.load('./../src/worldmodels1/memory_model.pth', map_location='cpu')
        # Remove 'module.' prefix from state_dict keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.rnn.load_state_dict(new_state_dict)
        self.rnn.eval()

        # check for cuda
        self.device = device
        if torch.cuda.is_available():
            self.vae = self.vae.cuda()
            self.rnn = self.rnn.cuda()
            self.device = 'cuda'
        
        print(f"Using device: {self.device}")


    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        #print("Shape of observation returned by reset:", obs.shape)
        self.hidden = (torch.zeros((1, 1, self.rnn.n_hidden)).to(self.device), 
               torch.zeros((1, 1, self.rnn.n_hidden)).to(self.device))
        # action should be a 1x3 tensor
        action = torch.zeros((1, 3)).to(self.device)
        obs = self.process_obs(obs, action)
        # if obs is a tensor, convert to numpy array
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()
        return obs, _

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = self.process_obs(obs, action)
        # if obs is a tensor, convert to numpy array
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()
        return obs, reward, done, trunc, info

    def process_obs(self, obs, action):
        # obs is 96x96x3 need to convert to 64x64x1. 
        # Convert to PyTorch tensor and normalize and permute dimensions and transfer to device
        obs = torch.from_numpy(obs).permute(2, 0, 1).float().to(self.device) / 255.0  
        # Rescale and average color channels
        obs = torch.nn.functional.interpolate(obs.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
        obs = obs.mean(dim=1, keepdim=True)  # Reduce color channels by taking the mean

        with torch.no_grad():
            #print(f'obs shape: {obs.shape}')
            mu, logvar = self.vae.encode(obs)
            z_t = self.vae.reparameterize(mu, logvar)
            #print(f'z_t shape: {z_t.shape}')
            # check if action is a tensor
            if not torch.is_tensor(action):
                action = torch.tensor(action).float().to(self.device).unsqueeze(0)
            rnn_in = torch.cat((z_t, action), dim=1).unsqueeze(0)
            #print(f"obs shape: {obs.shape}")
            #print(f"action shape: {action.shape}")
            #print(f"rnn_in shape: {rnn_in.shape}")
            #print(f"Initial hidden states shapes: {self.hidden[0].shape}, {self.hidden[1].shape}")
            _, _, _, self.hidden = self.rnn(rnn_in, self.hidden)
            # extract hidden state
            h_t = self.hidden[0]
            #print(f"Final hidden states shapes: {self.hidden[0].shape}, {self.hidden[1].shape}")
            #print(f"h_t shape: {h_t.shape}")
            #print(f"z_t shape: {z_t.shape}")
            # concat z_t and hidden
            z_t = z_t.squeeze(0) # remove batch dimension
            h_t = h_t.squeeze(0).squeeze(0) # remove batch and sequence dimensions
            obs = torch.cat((z_t, h_t))
            #print(f"obs shape: {obs.shape}")
        return obs