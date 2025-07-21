"""Simple script to run a trained PPO model on the image-based Chrome Dinosaur environment."""

from ChromeDinoImageEnv import ChromeDinoImageEnv
from stable_baselines3 import PPO
import time

def run_model(model_path: str = "ppo_chromedino_image.zip", max_steps: int = 1000):
    """Run a trained model on the image-based environment.
    
    Args:
        model_path: Path to the trained model
        max_steps: Maximum steps to run
    """
    # Create environment
    env = ChromeDinoImageEnv()
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Run the model
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        if done or truncated:
            print(f"Episode ended at step {step}, Total reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0
        
        time.sleep(0.05)  # Small delay for visibility
    
    env.close()

if __name__ == "__main__":
    run_model()
