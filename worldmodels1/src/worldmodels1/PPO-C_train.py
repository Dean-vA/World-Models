from stable_baselines3 import PPO
from gym_wrapper import CarRacingWrapper  
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    original_env = gym.make('CarRacing-v2')
    wrapped_env = CarRacingWrapper(original_env)
    return wrapped_env

envs = DummyVecEnv([make_env for _ in range(64)])  # Four parallel environments

model = PPO("MlpPolicy", envs, verbose=2)

# # Wrap the environment
# wrapped_env = CarRacingWrapper(original_env)

# # Initialize PPO model with a predefined policy (MlpPolicy)
# model = PPO("MlpPolicy", wrapped_env, verbose=2)

# Train the model in a loop saving checkpoints at every 20000 steps and continue training from the last saved checkpoint
for i in range(100):
    model.learn(total_timesteps=20000, reset_num_timesteps=False)
    model.save("ppo_car_racing_{}".format(i))
    print("Saved model checkpoint to {}".format("ppo_car_racing_{}".format(i)))

