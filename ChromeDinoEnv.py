import gymnasium as gym
from gymnasium import spaces
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    WebDriverException,
    JavascriptException
)
import time
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
GAME_URL = "https://chromedino.com/"
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 380
MAX_EPISODE_STEPS = 10000
DEFAULT_OBSTACLE_DISTANCE = 1000

class ChromeDinoEnv(gym.Env):
    """Chrome Dinosaur Game Environment for Reinforcement Learning.
    
    This environment interfaces with the Chrome Dinosaur game through Selenium WebDriver,
    providing observations about the game state and allowing the agent to control the dinosaur.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """Initialize the Chrome Dinosaur environment."""
        super().__init__()
        
        # Initialize WebDriver
        self.driver = self._create_webdriver()
        self._load_game()
        
        # Define observation and action spaces
        # Observation: [speed_ratio, obstacle_dist, obstacle_width, obstacle_height, 
        #               next_obstacle_dist, runner_y, runner_velocity, is_jumping]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, -20, 0], dtype=np.float32),
            high=np.array([2, 1200, 100, 100, 1200, 200, 20, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: jump
        
        # Environment state
        self.episode_steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.done = False
        self.last_observation = None
        self.game_started = False
        
        # Reward configuration
        self.survival_reward = 1
        self.crash_penalty = -10.0
        self.step_penalty = -0.01
        
        logger.info("ChromeDinoEnv initialized successfully")
    
    def _create_webdriver(self) -> webdriver.Chrome:
        """Create and configure Chrome WebDriver with optimal settings."""
        options = webdriver.ChromeOptions()
        
        # Performance and stability options
        options.add_argument('--disable-gpu')
        options.add_argument(f'--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}')
        options.add_argument('--hide-scrollbars')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-logging')
        options.add_argument('--log-level=3')
        options.add_argument('--disable-web-security')
        options.add_argument('--disable-features=VizDisplayCompositor')
        
        # Suppress automation detection
        options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        return webdriver.Chrome(options=options)
    
    def _load_game(self) -> None:
        """Load the Chrome Dinosaur game page."""
        try:
            self.driver.get(GAME_URL)
            # Wait for page to load completely
            WebDriverWait(self.driver, 10).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            time.sleep(2)  # Additional wait for game initialization
            logger.info(f"Game loaded successfully from {GAME_URL}")
        except Exception as e:
            logger.error(f"Failed to load game: {e}")
            raise
    
    def _start_game(self) -> None:
        """Start the Chrome Dinosaur game."""
        try:
            # Wait for the page to load and body element to be present
            wait = WebDriverWait(self.driver, 10)
            body = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            
            # Wait for game to be ready
            self._wait_for_game_ready()
            
            # Start the game
            body.send_keys(Keys.SPACE)
            time.sleep(0.5)
            self.game_started = True
            logger.info("Game started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start game: {e}")
            # Fallback: try JavaScript approach
            try:
                self.driver.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {'keyCode': 32}));")
                time.sleep(0.5)
                self.game_started = True
                logger.info("Game started using JavaScript fallback")
            except Exception as js_error:
                logger.error(f"JavaScript fallback also failed: {js_error}")
                raise
    
    def _wait_for_game_ready(self) -> None:
        """Wait for the game to be fully loaded and ready."""
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                # Check if Runner object exists
                game_ready = self.driver.execute_script(
                    "return typeof Runner !== 'undefined' && Runner.instance_ !== null;"
                )
                if game_ready:
                    logger.info("Game is ready")
                    return
            except Exception:
                pass
            
            time.sleep(0.5)
        
        raise TimeoutException("Game failed to load within expected time")
    
    def _get_observation(self) -> np.ndarray:
        """Get the current game state observation.
        
        Returns:
            np.ndarray: Observation vector containing game state information
        """
        try:
            # JavaScript to extract game state
            js_code = """
            try {
                if (typeof Runner === 'undefined' || !Runner.instance_) {
                    return null;
                }
                
                var runner = Runner.instance_;
                var trex = runner.tRex;
                
                // Get obstacles
                var obstacles = [];
                if (runner.horizon && runner.horizon.obstacles) {
                    obstacles = runner.horizon.obstacles
                        .filter(function(obs) { return obs.xPos > trex.xPos; })
                        .map(function(obs) {
                            return {
                                xPos: obs.xPos - trex.xPos,
                                width: obs.width || 0,
                                height: (obs.typeConfig && obs.typeConfig.height) || 0
                            };
                        })
                        .slice(0, 2);  // Only get first 2 obstacles
                }
                
                // Ensure we have at least 2 obstacle entries
                while (obstacles.length < 2) {
                    obstacles.push({xPos: 1000, width: 0, height: 0});
                }
                
                return {
                    speed: runner.currentSpeed || 0,
                    obstacle_dist: obstacles[0].xPos,
                    obstacle_width: obstacles[0].width,
                    obstacle_height: obstacles[0].height,
                    next_obstacle_dist: obstacles[1].xPos,
                    runner_y: trex.yPos || 0,
                    runner_velocity: trex.jumpVelocity || 0,
                    is_jumping: trex.jumping ? 1 : 0
                };
            } catch (error) {
                console.error('Error in observation script:', error);
                return null;
            }
            """
            
            # Execute JavaScript and get observation
            raw_obs = self.driver.execute_script(js_code)
            
            if raw_obs is None:
                logger.warning("JavaScript returned null observation, using default values")
                return self._get_default_observation()
            
            # Calculate speed ratio for better normalization
            speed_ratio = min(raw_obs['speed'] / 10.0, 2.0) if raw_obs['speed'] > 0 else 0.0
            
            # Create normalized observation vector
            observation = np.array([
                speed_ratio,
                min(raw_obs['obstacle_dist'], 1200),
                min(raw_obs['obstacle_width'], 100),
                min(raw_obs['obstacle_height'], 100),
                min(raw_obs['next_obstacle_dist'], 1200),
                min(raw_obs['runner_y'], 200),
                max(-20, min(raw_obs['runner_velocity'], 20)),
                raw_obs['is_jumping']
            ], dtype=np.float32)
            
            return observation
            
        except (WebDriverException, JavascriptException) as e:
            logger.error(f"WebDriver error in _get_observation: {e}")
            return self._get_default_observation()
        except Exception as e:
            logger.error(f"Unexpected error in _get_observation: {e}")
            return self._get_default_observation()
    
    def _get_default_observation(self) -> np.ndarray:
        """Return a safe default observation when game state cannot be read."""
        return np.array([0.0, DEFAULT_OBSTACLE_DISTANCE, 0.0, 0.0, DEFAULT_OBSTACLE_DISTANCE, 0.0, 0.0, 0.0], dtype=np.float32)


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take (0: do nothing, 1: jump)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            return self.last_observation, 0.0, True, False, {}

        # Execute action
        if action == 1:
            self._perform_jump()
        
        # Small delay to allow game to update
        time.sleep(0.05)
        
        # Update step counter
        self.episode_steps += 1
        
        # Get current observation and check game state
        obs = self._get_observation()
        crashed = self._is_game_crashed()
        
        # Calculate reward
        reward = self._calculate_reward(crashed)
        
        # Determine if episode is finished
        terminated = crashed
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Prepare info dictionary
        info = self._get_step_info(terminated, truncated)
        
        # Update environment state
        self.done = terminated or truncated
        self.last_observation = obs
        
        return obs, reward, terminated, truncated, info
    
    def _perform_jump(self) -> None:
        """Execute a jump action in the game."""
        try:
            body = self.driver.find_element(By.TAG_NAME, 'body')
            body.send_keys(Keys.SPACE)
        except NoSuchElementException:
            # Fallback to JavaScript
            try:
                self.driver.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {'keyCode': 32}));")
            except Exception as e:
                logger.warning(f"Failed to perform jump action: {e}")
    
    def _is_game_crashed(self) -> bool:
        """Check if the game has crashed (dinosaur hit an obstacle)."""
        try:
            crashed = self.driver.execute_script(
                "return (typeof Runner !== 'undefined' && Runner.instance_) ? Runner.instance_.crashed : false;"
            )
            return bool(crashed)
        except Exception as e:
            logger.warning(f"Failed to check crash status: {e}")
            return False
    
    def _calculate_reward(self, crashed: bool) -> float:
        """Calculate reward for the current step.
        
        Args:
            crashed: Whether the game crashed this step
            
        Returns:
            Calculated reward value
        """
        if crashed:
            return self.crash_penalty
        
        # Survival reward increases slightly with time to encourage longer runs
        survival_bonus = self.survival_reward * (1 + self.episode_steps * 0.0001)
        return survival_bonus + self.step_penalty
    
    def _get_step_info(self, terminated: bool, truncated: bool) -> Dict[str, Any]:
        """Get information dictionary for the step.
        
        Args:
            terminated: Whether episode terminated due to crash
            truncated: Whether episode was truncated due to max steps
            
        Returns:
            Info dictionary
        """
        info = {
            'episode_steps': self.episode_steps,
            'crashed': terminated,
            'truncated': truncated
        }
        
        if terminated or truncated:
            final_score = self.episode_steps * self.survival_reward
            info['final_score'] = final_score
            info['episode_length'] = self.episode_steps
            
            logger.info(
                f"Episode finished: steps={self.episode_steps}, "
                f"score={final_score:.2f}, crashed={terminated}, truncated={truncated}"
            )
        
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (initial_observation, info)
        """
        super().reset(seed=seed)
        
        # Reset environment state
        self.episode_steps = 0
        self.done = False
        self.game_started = False
        
        try:
            # Restart the game
            self._restart_game()
            
            # Get initial observation
            obs = self._get_observation()
            self.last_observation = obs
            
            info = {'episode_steps': 0}
            logger.info("Environment reset successfully")
            
            return obs, info
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            # Return safe default observation
            obs = self._get_default_observation()
            self.last_observation = obs
            return obs, {'reset_error': str(e)}
    
    def _restart_game(self) -> None:
        """Restart the game by refreshing or restarting."""
        try:
            # Try to restart using JavaScript first (faster)
            restart_success = self.driver.execute_script("""
                try {
                    if (typeof Runner !== 'undefined' && Runner.instance_) {
                        Runner.instance_.restart();
                        return true;
                    }
                    return false;
                } catch (error) {
                    return false;
                }
            """)
            
            if restart_success:
                time.sleep(0.5)
                logger.info("Game restarted using JavaScript")
            else:
                # Fallback: reload the page
                logger.info("JavaScript restart failed, reloading page")
                self.driver.refresh()
                self._load_game()
                self._start_game()
                
        except Exception as e:
            logger.error(f"Failed to restart game: {e}")
            # Last resort: reload page
            self.driver.refresh()
            self._load_game()
            self._start_game()

    def render(self, mode: str = 'human') -> None:
        """Render the environment (no-op since game is visually rendered in browser).
        
        Args:
            mode: Render mode (unused)
        """
        # The game is already visually rendered in the browser window
        pass

    def close(self) -> None:
        """Clean up resources and close the environment."""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {e}")
        finally:
            self.driver = None
