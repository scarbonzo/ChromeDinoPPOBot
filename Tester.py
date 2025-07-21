"""Chrome Dinosaur Environment Tester.

This script tests the ChromeDinoEnv environment by running random actions
and displaying the observations, rewards, and episode information.
"""

import time
import logging
import numpy as np
from ChromeDinoEnv import ChromeDinoEnv
from ChromeDinoImageEnv import ChromeDinoImageEnv
from stable_baselines3 import PPO
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment(max_steps: int = 100, delay: float = 0.1, use_images: bool = False):
    """Test the Chrome Dinosaur environment with random actions.
    
    Args:
        max_steps: Maximum number of steps to run
        delay: Delay between steps in seconds
        use_images: Whether to use image-based environment
    """
    env_type = "image-based" if use_images else "state-based"
    logger.info(f"Starting {env_type} environment test with random actions")
    
    try:
        # Create environment
        env = ChromeDinoImageEnv() if use_images else ChromeDinoEnv()
        
        # Reset environment
        obs, info = env.reset()
        logger.info(f"Environment reset. Initial observation shape: {obs.shape}")
        logger.info(f"Initial observation: {obs}")
        
        # Run episode
        step = 0
        total_reward = 0
        
        while step < max_steps:
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Log step information
            action_name = "JUMP" if action == 1 else "WAIT"
            logger.info(
                f"Step {step:3d}: Action={action_name}, Reward={reward:6.3f}, "
                f"Total={total_reward:7.3f}, Done={terminated or truncated}"
            )
            
            if step % 10 == 0:
                logger.info(f"  Observation: {obs}")
            
            step += 1
            
            # Check if episode ended
            if terminated or truncated:
                reason = "crashed" if terminated else "truncated"
                logger.info(f"Episode ended after {step} steps ({reason})")
                logger.info(f"Final info: {info}")
                break
            
            # Delay for visualization
            time.sleep(delay)
        
        logger.info(f"Test completed. Total steps: {step}, Total reward: {total_reward:.3f}")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()
            logger.info("Environment closed")

def test_trained_model(model_path: str, max_steps: int = 500, use_images: bool = False):
    """Test a trained PPO model on the Chrome Dinosaur game.
    
    Args:
        model_path: Path to the trained model
        max_steps: Maximum number of steps to run
        use_images: Whether to use image-based environment
    """
    env_type = "image-based" if use_images else "state-based"
    logger.info(f"Testing {env_type} trained model from {model_path}")
    
    try:
        # Create environment
        env = ChromeDinoImageEnv() if use_images else ChromeDinoEnv()
        
        # Load trained model
        model = PPO.load(model_path)
        logger.info("Model loaded successfully")
        
        # Reset environment
        obs, info = env.reset()
        
        # Run episode with trained model
        step = 0
        total_reward = 0
        
        while step < max_steps:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Log step information
            action_name = "JUMP" if action == 1 else "WAIT"
            if step % 50 == 0:
                logger.info(
                    f"Step {step:3d}: Action={action_name}, Reward={reward:6.3f}, "
                    f"Total={total_reward:7.3f}"
                )
            
            step += 1
            
            # Check if episode ended
            if terminated or truncated:
                reason = "crashed" if terminated else "truncated"
                logger.info(f"Episode ended after {step} steps ({reason})")
                logger.info(f"Final info: {info}")
                break
            
            time.sleep(0.05)  # Faster for trained model
        
        logger.info(f"Model test completed. Steps: {step}, Total reward: {total_reward:.3f}")
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()
            logger.info("Environment closed")

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test Chrome Dinosaur environment")
    parser.add_argument('--mode', choices=['random', 'model'], default='random',
                       help='Test mode: random actions or trained model')
    parser.add_argument('--env-type', choices=['state', 'image'], default='state',
                       help='Environment type: state-based or image-based')
    parser.add_argument('--model-path', type=str, default='ppo_chromedino.zip',
                       help='Path to trained model (for model mode)')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum number of steps to run')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between steps in seconds')
    
    args = parser.parse_args()
    use_images = args.env_type == 'image'
    
    if args.mode == 'random':
        test_environment(args.max_steps, args.delay, use_images)
    else:
        test_trained_model(args.model_path, args.max_steps, use_images)

if __name__ == "__main__":
    main()
