from stable_baselines3 import PPO
from gym_wrapper import CarRacingWrapper  
import gymnasium as gym

# Initialize the original environment
original_env = gym.make('CarRacing-v2')  # Replace with the name of your original environment

# Wrap the environment
wrapped_env = CarRacingWrapper(original_env)

# Initialize PPO model with a predefined policy (MlpPolicy)
model = PPO("MlpPolicy", wrapped_env, verbose=2)

# Train the model in a loop saving checkpoints at every 20000 steps and continue training from the last saved checkpoint
for i in range(100):
    model.learn(total_timesteps=20000, reset_num_timesteps=False)
    model.save("ppo_car_racing_{}".format(i))

