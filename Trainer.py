from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from ChromeDinoEnv import ChromeDinoEnv
import numpy as np
import argparse
import os

#Hyperparameters
TIMESTEPS = 1000 # Number of timesteps to train for
NUM_ENVS = 1  # Adjust based on your system capability
N_STEPS = 256 # Number of steps to run before updating the policy
BATCH_SIZE = 256 # Number of steps to collect before updating the policy
N_EPOCHS = 5 # Number of epochs to run before updating the policy
LEARNING_RATE = 5e-4 # Learning rate for the policy
ENT_COEF = 0.05 # Entropy coefficient
CLIP_RANGE = 0.2 # Clip range for the policy
CLIP_RANGE_VF = 0.1 # Clip range for the value function
GAMMA = 0.99 # Discount factor
POLICY_KWARGS = dict(net_arch=dict(pi=[64,64], vf=[128,128])) # Policy network architecture

#Training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from saved model')
    args = parser.parse_args()

    num_envs = NUM_ENVS
    env = SubprocVecEnv([ChromeDinoEnv for i in range(num_envs)])

    model_path = "ppo_chromedino.zip"
    if args.resume and os.path.exists(model_path):
        print("[INFO] Resuming training from saved model...")
        model = PPO.load(model_path, env=env)
    else:
        print("[INFO] Starting new model training...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
            clip_range_vf=CLIP_RANGE_VF,
            gamma=GAMMA,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log="./tensorboard/"
        )

    model.learn(total_timesteps=TIMESTEPS)
    model.save("ppo_chromedino")
    env.close()

    total_timesteps = model.num_timesteps
    ep_rewards = [ep_info['r'] for ep_info in model.ep_info_buffer]
    mean_reward = np.mean(ep_rewards) if ep_rewards else float('nan')
    print(f"Training complete: total_timesteps={total_timesteps}, mean_reward={mean_reward:.2f}")
    print("PPO training complete. Model saved as 'ppo_chromedino.zip'.")
