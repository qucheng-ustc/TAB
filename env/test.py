import gym

env = gym.make('eth2-v3')
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
env.reset()
