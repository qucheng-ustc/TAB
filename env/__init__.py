import gym
from gym.envs.registration import register
from env.eth2 import Eth2, Eth2v1, Eth2v2, Eth2v3

register(
        id='eth2-v0',
        entry_point='env:Eth2'
)

register(
        id='eth2-v1',
        entry_point='env:Eth2v1'
)

register(
        id='eth2-v2',
        entry_point='env:Eth2v2'
)

register(
        id='eth2-v3',
        entry_point='env:Eth2v3'
)

