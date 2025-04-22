import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from Environments.environments import EnvironmentStock

env = EnvironmentStock()
save_path = os.path.join("Training", "Saved Models", "PPO_perish_full_information")
model_input_string = "MlpPolicy"
load_model = False
train_model = True
evaluate_model = True
test_model_on_environment = False

#for faster training
env = DummyVecEnv([lambda: env])

if load_model:
    model = PPO.load(save_path, env=env)
else:
    model = PPO(model_input_string, env, verbose=1, tensorboard_log="./Training/Logs/PPO_perish_log")

# # Train Model
if train_model:
    model.learn(total_timesteps=5e5) 

# # Save Model
model.save(save_path)


# # Evaluate Model
if evaluate_model:
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# Test Model
if test_model_on_environment:
    episodes = 5
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
            print(f"Action: {action}, Obs: {obs}, Reward: {reward}")
            print("__________________________")

        print(f"Episode: {episode}, Score: {score}")
