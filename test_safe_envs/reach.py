import gym
import panda_gym
import time
env = gym.make("PandaReach-v2", render=True)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render(mode='human')
    # time.sleep(9)


env.close()
