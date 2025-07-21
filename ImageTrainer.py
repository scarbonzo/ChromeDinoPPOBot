"""Chrome Dinosaur Image-Based PPO Training Script.

This script trains a PPO agent to play the Chrome Dinosaur game using frame-stacked
images as observations instead of JavaScript-extracted game state values.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from ChromeDinoImageEnv import ChromeDinoImageEnv
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

# Training Configuration for Image-Based Environment
class ImageTrainingConfig:
    """Configuration class for image-based training hyperparameters."""
    
    # Training parameters
    TIMESTEPS = 500_000  # Number of timesteps to train for
    NUM_ENVS = 1  # Number of parallel environments (keep at 1 for Chrome game)
    
    # PPO hyperparameters (adjusted for image observations)
    N_STEPS = 1024  # Larger steps for image-based learning
    BATCH_SIZE = 64  # Batch size for training
    N_EPOCHS = 4  # Fewer epochs for image data
    LEARNING_RATE = 2.5e-4  # Lower learning rate for CNN
    ENT_COEF = 0.02  # Entropy coefficient (encourage exploration)
    CLIP_RANGE = 0.2  # Smaller clip range for stability
    CLIP_RANGE_VF = 0.1  # Clip range for the value function
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE lambda parameter
    
    # CNN Network architecture for image processing
    POLICY_KWARGS = {
        'net_arch': dict(pi=[256], vf=[256]),
        'activation_fn': nn.ReLU
    }
    
    # File paths
    MODEL_PATH = "ppo_chromedino_image"
    TENSORBOARD_LOG = "./tensorboard_image/"
    
    # Evaluation
    EVAL_FREQ = 5000
    EVAL_EPISODES = 3


def create_environment() -> DummyVecEnv:
    """Create and wrap the Chrome Dinosaur image environment.
    
    Returns:
        Vectorized environment ready for training
    """
    def make_env():
        return ChromeDinoImageEnv()
    
    # Use DummyVecEnv for better stability with browser automation
    env = DummyVecEnv([make_env])
    logger.info("Image environment created successfully")
    return env


def create_model(env: DummyVecEnv, resume_path: Optional[str] = None) -> PPO:
    """Create or load PPO model for image-based training.
    
    Args:
        env: The environment to train on
        resume_path: Path to existing model to resume from
        
    Returns:
        PPO model ready for training
    """
    config = ImageTrainingConfig()
    
    if resume_path and os.path.exists(resume_path + ".zip"):
        logger.info(f"Loading existing model from {resume_path}.zip")
        model = PPO.load(resume_path, env=env)
    else:
        logger.info("Creating new PPO model for image-based training")
        model = PPO(
            "CnnPolicy",  # Use CNN policy for image observations
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
    """Main training function for image-based environment."""
    parser = argparse.ArgumentParser(description="Train PPO agent on Chrome Dinosaur game using images")
    parser.add_argument('--resume', action='store_true', help='Resume training from saved model')
    parser.add_argument('--timesteps', type=int, default=ImageTrainingConfig.TIMESTEPS, 
                       help='Number of timesteps to train for')
    parser.add_argument('--model-path', type=str, default=ImageTrainingConfig.MODEL_PATH,
                       help='Path to save/load model')
    args = parser.parse_args()
    
    logger.info("Starting Chrome Dinosaur Image-Based PPO training")
    logger.info(f"Training timesteps: {args.timesteps}")
    logger.info("Using frame-stacked images as observations")
    
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
        logger.info("Image-based training completed successfully")
        
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
