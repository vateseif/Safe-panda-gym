import gym
import panda_gym
import time

env = gym.make("PandaPushSafe-v2", render=True)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cost = info['cost']
    # print("cost", cost)
    # print("reward" ,  reward)
    # print(obs['observation'].size)
    env.render(mode='human')
    time.sleep(2)


env.close()
