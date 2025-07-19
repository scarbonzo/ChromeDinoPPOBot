# Chrome Dinosaur PPO Bot

A reinforcement learning agent that learns to play the Chrome Dinosaur game using Proximal Policy Optimization (PPO) from Stable Baselines3.

## Overview

This project implements a PPO agent that controls the Chrome Dinosaur (T-Rex) game through browser automation using Selenium WebDriver. The agent learns to jump over obstacles to maximize survival time and score.

## Features

- **Robust Environment**: ChromeDinoEnv with comprehensive error handling and fallback mechanisms
- **Optimized PPO Training**: Fine-tuned hyperparameters for stable learning
- **Browser Automation**: Selenium WebDriver integration with Chrome browser
- **Comprehensive Testing**: Tools for testing both random actions and trained models
- **Logging & Monitoring**: Detailed logging and TensorBoard integration
- **Type Safety**: Full type hints throughout the codebase

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ChromeDinoPPOBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install ChromeDriver:
   - Download ChromeDriver from https://chromedriver.chromium.org/
   - Ensure ChromeDriver is in your PATH or place it in the project directory

## Usage

### Training a New Model

Train a PPO agent from scratch:
```bash
python Trainer.py
```

With custom parameters:
```bash
python Trainer.py --timesteps 50000 --model-path my_model
```

### Resume Training

Resume training from a saved model:
```bash
python Trainer.py --resume --model-path ppo_chromedino
```

### Testing the Environment

Test with random actions:
```bash
python Tester.py --mode random --max-steps 200
```

Test a trained model:
```bash
python Tester.py --mode model --model-path ppo_chromedino.zip --max-steps 1000
```

## Environment Details

### Observation Space

The environment provides an 8-dimensional observation vector:
- `speed_ratio`: Normalized game speed (0-2)
- `obstacle_dist`: Distance to nearest obstacle (0-1200)
- `obstacle_width`: Width of nearest obstacle (0-100)
- `obstacle_height`: Height of nearest obstacle (0-100)
- `next_obstacle_dist`: Distance to second obstacle (0-1200)
- `runner_y`: Dinosaur's Y position (0-200)
- `runner_velocity`: Dinosaur's jump velocity (-20 to 20)
- `is_jumping`: Whether dinosaur is jumping (0 or 1)

### Action Space

- `0`: Do nothing (continue running)
- `1`: Jump

### Reward Structure

- **Survival reward**: +0.1 per step (increases slightly over time)
- **Step penalty**: -0.01 per step (encourages efficiency)
- **Crash penalty**: -10.0 when hitting an obstacle

## Configuration

### Training Hyperparameters

The `TrainingConfig` class in `Trainer.py` contains all hyperparameters:

```python
class TrainingConfig:
    TIMESTEPS = 10000        # Training timesteps
    N_STEPS = 512           # Steps per update
    BATCH_SIZE = 64         # Batch size
    N_EPOCHS = 10           # Training epochs per update
    LEARNING_RATE = 3e-4    # Learning rate
    ENT_COEF = 0.01         # Entropy coefficient
    GAMMA = 0.99            # Discount factor
    # ... more parameters
```

### Environment Configuration

Key environment settings in `ChromeDinoEnv.py`:

```python
GAME_URL = "https://chromedino.com/"
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
MAX_EPISODE_STEPS = 10000
```

## Architecture

### Files Structure

- `ChromeDinoEnv.py`: Custom Gymnasium environment for the Chrome Dinosaur game
- `Trainer.py`: PPO training script with optimized hyperparameters
- `Tester.py`: Testing utilities for environment and trained models
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

### Key Components

1. **ChromeDinoEnv**: 
   - Manages browser automation
   - Extracts game state observations
   - Handles rewards and episode termination
   - Robust error handling and recovery

2. **Training Pipeline**:
   - PPO algorithm from Stable Baselines3
   - DummyVecEnv for stable browser automation
   - TensorBoard logging for monitoring

3. **Testing Framework**:
   - Random action testing
   - Trained model evaluation
   - Comprehensive logging

## Troubleshooting

### Common Issues

1. **ChromeDriver not found**:
   - Ensure ChromeDriver is installed and in PATH
   - Check Chrome browser version compatibility

2. **JavaScript execution errors**:
   - The environment includes fallback mechanisms
   - Check browser console for game loading issues

3. **Training instability**:
   - Reduce learning rate or batch size
   - Increase entropy coefficient for more exploration

### Performance Tips

- Use `DummyVecEnv` instead of `SubprocVecEnv` for better browser stability
- Adjust `MAX_EPISODE_STEPS` based on your training goals
- Monitor TensorBoard logs for training progress

## Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir ./tensorboard/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper type hints and documentation
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stable Baselines3 for the PPO implementation
- Selenium WebDriver for browser automation
- Chrome Dinosaur game developers at Google
