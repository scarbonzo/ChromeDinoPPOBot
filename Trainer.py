"""Chrome Dinosaur PPO Training Script.

This script trains a PPO agent to play the Chrome Dinosaur game using Stable Baselines3.
The agent learns to jump over obstacles to maximize survival time.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from ChromeDinoEnv import ChromeDinoEnv
import numpy as np
import argparse
import os
import logging
from typing import Optional
import time
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Training Configuration
class TrainingConfig:
    """Configuration class for training hyperparameters."""
    
    # Training parameters
    TIMESTEPS = 500000  # Number of timesteps to train for
    NUM_ENVS = 1  # Number of parallel environments (keep at 1 for Chrome game)
    
    # PPO hyperparameters
    N_STEPS = 512  # Number of steps to run before updating the policy
    BATCH_SIZE = 64  # Batch size for training
    N_EPOCHS = 10  # Number of epochs to run before updating the policy
    LEARNING_RATE = 3e-4  # Learning rate for the policy
    ENT_COEF = 0.025  # Entropy coefficient (encourage exploration)
    CLIP_RANGE = 0.25  # Clip range for the policy
    CLIP_RANGE_VF = 0.25  # Clip range for the value function
    GAMMA = 0.995  # Discount factor
    GAE_LAMBDA = 0.95  # GAE lambda parameter
    
    # Network architecture
    POLICY_KWARGS = {
        'net_arch': dict(pi=[64, 64], vf=[256, 256]),
        'activation_fn': nn.Tanh
    }
    
    # File paths
    MODEL_PATH = "ppo_chromedino"
    TENSORBOARD_LOG = "./tensorboard/"
    
    # Evaluation
    EVAL_FREQ = 2000
    EVAL_EPISODES = 5

def create_environment() -> DummyVecEnv:
    """Create and wrap the Chrome Dinosaur environment.
    
    Returns:
        Vectorized environment ready for training
    """
    def make_env():
        return ChromeDinoEnv()
    
    # Use DummyVecEnv instead of SubprocVecEnv for better stability with browser automation
    env = DummyVecEnv([make_env])
    logger.info("Environment created successfully")
    return env

def create_model(env: DummyVecEnv, resume_path: Optional[str] = None) -> PPO:
    """Create or load PPO model.
    
    Args:
        env: The environment to train on
        resume_path: Path to existing model to resume from
        
    Returns:
        PPO model ready for training
    """
    config = TrainingConfig()
    
    if resume_path and os.path.exists(resume_path + ".zip"):
        logger.info(f"Loading existing model from {resume_path}.zip")
        model = PPO.load(resume_path, env=env)
    else:
        logger.info("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            ent_coef=config.ENT_COEF,
            clip_range=config.CLIP_RANGE,
            clip_range_vf=config.CLIP_RANGE_VF,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            policy_kwargs=config.POLICY_KWARGS,
            tensorboard_log=config.TENSORBOARD_LOG
        )
    
    return model

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO agent on Chrome Dinosaur game")
    parser.add_argument('--resume', action='store_true', help='Resume training from saved model')
    parser.add_argument('--timesteps', type=int, default=TrainingConfig.TIMESTEPS, 
                       help='Number of timesteps to train for')
    parser.add_argument('--model-path', type=str, default=TrainingConfig.MODEL_PATH,
                       help='Path to save/load model')
    args = parser.parse_args()
    
    logger.info("Starting Chrome Dinosaur PPO training")
    logger.info(f"Training timesteps: {args.timesteps}")
    
    try:
        # Create environment
        env = create_environment()
        
        # Create model
        model = create_model(env, args.model_path if args.resume else None)
        
        # Train the model
        logger.info("Starting training...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model
        model.save(args.model_path)
        logger.info(f"Model saved as {args.model_path}.zip")
        
        # Print training statistics
        total_timesteps = model.num_timesteps
        ep_rewards = [ep_info['r'] for ep_info in model.ep_info_buffer]
        mean_reward = np.mean(ep_rewards) if ep_rewards else float('nan')
        
        logger.info(f"Training Statistics:")
        logger.info(f"  Total timesteps: {total_timesteps}")
        logger.info(f"  Mean episode reward: {mean_reward:.2f}")
        logger.info(f"  Training time: {training_time:.2f}s")
        
        # Clean up
        env.close()
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'env' in locals():
            env.close()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if 'env' in locals():
            env.close()
        raise

if __name__ == "__main__":
    main()
