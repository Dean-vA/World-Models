from stable_baselines3 import PPO
from gym_wrapper import CarRacingWrapper  
import gymnasium as gym

# Initialize the original environment
original_env = gym.make('CarRacing-v2')  # Replace with the name of your original environment

# Wrap the environment
wrapped_env = CarRacingWrapper(original_env)

# Initialize PPO model with a predefined policy (MlpPolicy)
model = PPO("MlpPolicy", wrapped_env, verbose=2)

# Train the model
model.learn(total_timesteps=20000)

model.save("ppo_car_racing")


