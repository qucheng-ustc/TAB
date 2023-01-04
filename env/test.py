import gym

env = gym.make('eth2-v1')
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
env.reset()
