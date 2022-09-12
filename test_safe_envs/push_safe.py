import gym
import panda_gym
import time

env = gym.make("PandaPushSafe-v2", render=True)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, cost_reward, done, info = env.step(action)
    # print(obs['observation'].size)
    print(cost_reward)
    env.render(mode='human')
    time.sleep(2)


env.close()