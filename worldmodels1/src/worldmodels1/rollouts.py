import gym
import numpy as np
import argparse
from multiprocessing import Pool, current_process
import logging
from PIL import Image
import torch
from stable_baselines3 import PPO



def preprocess_state(state, img_size=64, gray_scale=False):
    # Convert the NumPy array to a PIL image
    img = Image.fromarray(state.astype('uint8'), 'RGB')
    # Resize the image using Pillow
    img = img.resize((img_size, img_size), Image.LANCZOS)
    if gray_scale:
        img = img.convert('L')
    # Convert the PIL image back to a NumPy array
    state = np.array(img).astype('uint8')
    
    return state

# a function to process the observation for the provided controller if any
def process_obs(obs, action, vae, rnn, hidden, device='cpu'):
    # obs is 96x96x3 need to convert to 64x64x1. 
    # Convert to PyTorch tensor and normalize and permute dimensions and transfer to 
    obs = torch.from_numpy(obs).permute(2, 0, 1).float().to(device) / 255.0  
    # Rescale and average color channels
    obs = torch.nn.functional.interpolate(obs.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
    obs = obs.mean(dim=1, keepdim=True)  # Reduce color channels by taking the mean

    with torch.no_grad():
        logging.info(f'obs shape: {obs.shape}')
        mu, logvar = vae.encode(obs)
        z_t = vae.reparameterize(mu, logvar)
        logging.info(f'z_t shape: {z_t.shape}')
        # check if action is a tensor
        if not torch.is_tensor(action):
            action = torch.tensor(action).float().to(device).unsqueeze(0)
        rnn_in = torch.cat((z_t, action), dim=1).unsqueeze(0)
        logging.info(f"obs shape: {obs.shape}")
        logging.info(f"action shape: {action.shape}")
        logging.info(f"rnn_in shape: {rnn_in.shape}")
        logging.info(f"Initial hidden states shapes: {hidden[0].shape}, {hidden[1].shape}")
        _, _, _, hidden = rnn(rnn_in, hidden)
        # extract hidden state
        h_t = hidden[0]
        logging.info(f"Final hidden states shapes: {hidden[0].shape}, {hidden[1].shape}")
        logging.info(f"h_t shape: {h_t.shape}")
        logging.info(f"z_t shape: {z_t.shape}")
        # concat z_t and hidden
        z_t = z_t.squeeze(0) # remove batch dimension
        h_t = h_t.squeeze(0).squeeze(0) # remove batch and sequence dimensions
        obs = torch.cat((z_t, h_t))
        #print(f"obs shape: {obs.shape}")
    return obs


def collect_data(env_name, num_episodes=10, max_steps=1000, seed=None, img_size=64, gray_scale=False, worldmodel=None):
    # print device that is being used by torch
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    worker_id = current_process()._identity[0]
    logging.info(f"Worker {worker_id}: Starting data collection for {num_episodes} episodes.")
    env = gym.make(env_name)
    if seed is not None:
        if hasattr(env, 'seed'):
            env.seed(seed + current_process()._identity[0])
    data = []
    
    for episode in range(num_episodes):
        logging.info(f"Worker {worker_id}: Starting episode {episode + 1}/{num_episodes}.")
        print(f"Worker {worker_id}: Starting episode {episode + 1}/{num_episodes}.")
        state = env.reset()[0]
        if worldmodel is None:
            state = preprocess_state(state, img_size=img_size, gray_scale=gray_scale)
        print(f'Worker {worker_id}: 1st state shape: {state.shape}, data type: {state.dtype}')
        done = False
        episode_data = []
        step_count = 0

        # Initialize the controller initial hidden state and action if provided
        if worldmodel is not None:
            print(f'Worker {worker_id}: initializing controller')
            hidden = (torch.zeros((1, 1, worldmodel['rnn'].n_hidden)), 
                      torch.zeros((1, 1, worldmodel['rnn'].n_hidden)))
            print(f'Worker {worker_id}: initializing action')
            action = torch.zeros((1, 3))
            print(f'Worker {worker_id}: loading controller')
            controller = PPO.load(worldmodel['controller_path'], device='cpu', verbose=1)  
            print(f'Worker {worker_id}: controller loaded')

        while not done and step_count < max_steps:
            logging.info(f"Worker {worker_id}: Starting step {step_count + 1}/{max_steps}.")
            print(f"Worker {worker_id}: Starting step {step_count + 1}/{max_steps}.")
            # Sample a random action from the environment's action space if no controller is provided
            if worldmodel is None:
                action = env.action_space.sample()
            else:
                obs = process_obs(state, action, worldmodel['vae'], worldmodel['rnn'], hidden)
                action, _ = controller.predict(obs)

            next_state, reward, done, truncated, info = env.step(action)
            episode_data.append((state, action, reward, done, episode, step_count)) #Step count and episode number to help with debugging
            # episode_data.append((state, action, reward, next_state, done, truncated, info))
            if worldmodel is None:
                state = preprocess_state(next_state, img_size=img_size, gray_scale=gray_scale) 
            else:
                state = next_state
            step_count += 1
            
        data.append(episode_data)
        logging.info(f"Worker {worker_id}: Finished episode {episode + 1}/{num_episodes}.")
        print(f"Worker {worker_id}: Finished episode {episode + 1}/{num_episodes} with {step_count} steps.")

    logging.info(f"Worker {worker_id}: Finished data collection.")
    print(f"Worker {worker_id}: Finished data collection.")
    env.close()
    return data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    parser = argparse.ArgumentParser(description="Collect data for CarRacing environment.")
    parser.add_argument("--env", default="CarRacing-v2", type=str, help="Name of the Gym environment")
    parser.add_argument("--workers", default=20, type=int, help="Number of worker processes")
    parser.add_argument("--episodes", default=5, type=int, help="Number of episodes per worker")
    parser.add_argument("--max_steps", default=1000, type=int, help="Max steps per episode")
    parser.add_argument("--seed", type=int, help="Base random seed")
    parser.add_argument("--img_size", default=64, type=int, help="Image size")
    parser.add_argument("--gray_scale", action="store_true", default=False, help="Convert images to grayscale")
    parser.add_argument("--output_path", default="collected_data.npy", type=str, help="Path to save the collected data")
    parser.add_argument("--use_controller", action="store_true", default=False, help="Use trained controller for rollouts")
    parser.add_argument("--controller_path", default="", type=str, help="Path to the trained controller model")
    args = parser.parse_args()


    worldmodel = None
    if args.use_controller:
        latent_dim = 32
        action_dim = 3
        from cnnvae import VAE
        from mdnrnn import MemoryModel
        if args.controller_path == "":
            print("Controller path must be specified when using a trained controller.")
            exit(1)
        
        vae = VAE()
        # Load the state_dict into CPU memory
        state_dict = torch.load('vae2.pth', map_location='cpu')
        # Remove 'module.' prefix from state_dict keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Load the modified state_dict into the model
        vae.load_state_dict(new_state_dict)
        vae.eval()
        
        rnn = MemoryModel(n_input=latent_dim+action_dim, n_hidden=256, n_gaussians=5, latent_dim=latent_dim)
        # Load the state_dict into CPU memory
        state_dict = torch.load('src/worldmodels1/memory_model.pth', map_location='cpu')
        # Remove 'module.' prefix from state_dict keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Load the modified state_dict into the model
        rnn.load_state_dict(new_state_dict)
        rnn.eval()
        
        controller_path = args.controller_path
        
        worldmodel = {'vae': vae, 'rnn': rnn, 'controller_path': controller_path}
    
    logging.info(f"Starting data collection for {args.episodes * args.workers} episodes.")
    with Pool(args.workers) as p:
        results = p.starmap(
            collect_data, 
            [(args.env, args.episodes, args.max_steps, args.seed, args.img_size, args.gray_scale, worldmodel) for _ in range(args.workers)]
        )

    collected_data = [item for sublist in results for item in sublist]
    np.save(args.output_path, np.array(collected_data, dtype=object))

