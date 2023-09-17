import gym
import numpy as np
import argparse
from multiprocessing import Pool, current_process
import logging
from PIL import Image


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


def collect_data(env_name, num_episodes=10, max_steps=1000, seed=None, img_size=64, gray_scale=False):
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
        state = preprocess_state(state, img_size=img_size, gray_scale=gray_scale)
        print(f'Worker {worker_id}: 1st state shape: {state.shape}, data type: {state.dtype}')
        done = False
        episode_data = []
        step_count = 0
        
        while not done and step_count < max_steps:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            episode_data.append((state, action, reward, done, truncated))
            # episode_data.append((state, action, reward, next_state, done, truncated, info))
            state = preprocess_state(next_state, img_size=img_size, gray_scale=gray_scale)
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
    args = parser.parse_args()
    
    logging.info(f"Starting data collection for {args.episodes * args.workers} episodes.")
    with Pool(args.workers) as p:
        results = p.starmap(
            collect_data, 
            [(args.env, args.episodes, args.max_steps, args.seed, args.img_size, args.gray_scale) for _ in range(args.workers)]
        )

    collected_data = [item for sublist in results for item in sublist]
    np.save("collected_data.npy", np.array(collected_data, dtype=object))

