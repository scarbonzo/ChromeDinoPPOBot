import time
from ChromeDinoEnv import ChromeDinoEnv

env = ChromeDinoEnv()
obs = env.reset()
print('Initial observation:', obs)

terminated = False
truncated = False
step = 0
while not (terminated or truncated) and step < 100:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}")
    step += 1
    time.sleep(0.1)
env.close()
