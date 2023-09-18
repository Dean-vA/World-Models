import argparse
from cnnvae import VAE
import torch
import numpy as np
from tqdm import tqdm
from utils import CarRacingDataset, get_dataloader
import logging

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('Starting the script...')
    
    data_path = args.data_path
    batch_size = args.batch_size
    num_workers = args.num_workers

    logging.info(f'Using data path: {data_path}, batch size: {batch_size}, number of workers: {num_workers}')

    logging.info(f'Loading data from {data_path}')
    try:
        preprocessed_data = np.load(data_path, allow_pickle=True)
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error while loading data: {e}')
        return

    logging.info('Initializing dataset and dataloader...')
    try:
        dataset = CarRacingDataset(preprocessed_data)
        dataloader = get_dataloader(preprocessed_data, batch_size, num_workers)
        logging.info('Dataset and dataloader initialized successfully.')
    except Exception as e:
        logging.error(f'Error while initializing dataset and dataloader: {e}')
        return

    logging.info('Loading VAE model...')
    try:
        vae = VAE()
        vae.load_state_dict(torch.load('vae.pth'))
        vae = vae.to(device="cuda")
        vae.eval()
        logging.info('VAE model loaded successfully.')
    except Exception as e:
        logging.error(f'Error while loading VAE model: {e}')
        return

    latent_action_pairs = []

    logging.info('Generating latent vectors...')
    try:
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc='Creating latent vectors', unit='batch') as pbar:
                for batch in dataloader:
                    states = torch.stack([torch.tensor(t[0], dtype=torch.float32).to(device="cuda") for t in batch])
                    actions = torch.stack([torch.tensor(t[1], dtype=torch.float32) for t in batch]).cpu().numpy()

                    mu, logvar = vae.encoder(states)
                    z = vae.reparameterize(mu, logvar)

                    latent_vectors = z.cpu().numpy()

                    for latent, action in zip(latent_vectors, actions):
                        latent_action_pairs.append(np.concatenate([latent, action]))
                    pbar.update(1)
        logging.info('Latent vectors generated successfully.')
    except Exception as e:
        logging.error(f'Error while generating latent vectors: {e}')
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate latent vectors using a pretrained VAE model.')
    parser.add_argument('--data_path', type=str, default='data/collected_data.npy', help='Path to the data file.')
    parser.add_argument('--vae_path', type=str, default='vae.pth', help='Path to the pretrained VAE model.')
    parser.add_argument('--output_path', type=str, default='data/latent_action_pairs.npy', help='Path to the output file.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the dataloader.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for the dataloader.')
    
    args = parser.parse_args()
    
    main(args)
