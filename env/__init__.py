import gym
from gym.envs.registration import register
from env.eth2 import Eth2, Eth2v1, Eth2v2, Eth2v3, Eth2v4, Eth2v5, Eth2v301, Eth2v302

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

register(
        id='eth2-v301',
        entry_point='env:Eth2v301'
)

register(
        id='eth2-v302',
        entry_point='env:Eth2v302'
)

register(
        id='eth2-v4',
        entry_point='env:Eth2v4'
)

register(
        id='eth2-v5',
        entry_point='env:Eth2v5'
)