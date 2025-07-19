import gymnasium as gym
from gymnasium import spaces
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# Hyperparameters
#GAME_URL = "https://dino-chrome.com/"
GAME_URL = "https://chromedino.com/"
WINDOW_WIDTH  = 1000
WINDOW_HEIGHT = 1000

class ChromeDinoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        options.add_argument(f'--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}')
        options.add_argument('--hide-scrollbars')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-logging')
        options.add_argument('--log-level=3')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option('useAutomationExtension', False)
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(GAME_URL)
        time.sleep(2)
        self._start_game()

        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        low  = np.array([ 0,    0,   0,   0,   0,   0,   0,   0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(2)  # 0: nothing, 1: jump

        # bonus bookkeeping
        self.bonus_frac    = 0.1
        self.episode_steps = 0
        self.done          = False
        self.last_observation = None

    def _start_game(self):
        # Wait for the page to load and body element to be present
        wait = WebDriverWait(self.driver, 10)
        body = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        
        # Additional wait to ensure the game is fully loaded
        time.sleep(2)
        
        # Try to start the game
        body.send_keys(Keys.SPACE)
        time.sleep(0.5)
        
    def _get_observation(self):
        js = """
        var r     = Runner.instance_;
        var t     = r.tRex;
        var obsArr= (r.horizon && r.horizon.obstacles)
        ? r.horizon.obstacles.map(o=>({
            xPos: o.xPos,
                width: o.width,
                height: o.typeConfig?.height||0
                }))
            : [];
        var o0 = obsArr[0]||{xPos:1000,width:0,height:0};
        var o1 = obsArr[1]||{xPos:1000,width:0,height:0};
        return {
            speed:          r.currentSpeed,
            obstacle_dist:  o0.xPos,
            obstacle_width: o0.width,
            obstacle_height:o0.height,
            next_dist:      o1.xPos,
            runner_y:       t.yPos,
            runner_vel:     t.jumpVelocity,
            is_jumping:     t.jumping?1:0
        };
        """
        obs = self.driver.execute_script(js)
        if obs['speed'] > 0:
            speed = obs['obstacle_dist']/obs['speed']
        else:
            speed = 0
        return np.array([
            speed,
            obs['obstacle_dist'],
            obs['obstacle_width'],
            obs['obstacle_height'],
            obs['next_dist'],
            obs['runner_y'],
            speed,
            obs['is_jumping']
        ], dtype=np.float32)


    def step(self, action):
        if self.done:
            return self.last_observation, 0.0, True, False, {}

        # perform action
        if action == 1:
            body = self.driver.find_element(By.TAG_NAME, 'body')
            body.send_keys(Keys.SPACE)
        time.sleep(0.05)

        # count this step
        self.episode_steps += 1

        # get obs & check crash
        obs     = self._get_observation()
        crashed = self.driver.execute_script("return Runner.instance_.crashed;")

        # base reward
        reward = self.episode_steps if not crashed else -100.0

        # terminal flag
        terminated = bool(crashed)
        truncated  = False

        info = {}
        if terminated:
            bonus = (self.bonus_frac * self.episode_steps) * 5 # Really pump the rewards for running longer
            reward += bonus
            info['length_bonus'] = bonus
            print(f"Episode finished after {self.episode_steps} steps, bonus: {bonus:.2f}, total reward: {reward:.2f}")

        # save state
        self.done = terminated
        self.last_observation = obs

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(1)
        self._start_game()
        self.done          = False
        self.episode_steps = 0
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        obs = self._get_observation()
        self.last_observation = obs
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.driver.quit()
