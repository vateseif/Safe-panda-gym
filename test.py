import gym
import panda_gym
import time
import numpy as np
env = gym.make("PandaCookSteak", render=True)
obs = env.reset()
done = False
print(env.action_space)
while True:
    # actions = [s.sample() for s in env.action_space]
    actions = [np.zeros(4) for _ in range(2)]
    env.step(actions)
    time.sleep(0.05)
input("Press Enter to continue...")
env.close()