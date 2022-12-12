import gym
from gym.envs.registration import register
from env.eth2 import Eth2v1, Eth2v2

register(
        id='eth2-v1',
        entry_point='env:Eth2v1'
)

register(
        id='eth2-v2',
        entry_point='env:Eth2v2'
)
