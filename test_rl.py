# Import the RL algorithm (Algorithm) we would like to use.
import ray
from ray.rllib.algorithms.ppo import PPO
from env import Eth2v1
from arrl.dataset import Dataset
from strategy import GroupAllocateStrategy
import utils
utils.tfutil.set_gpu_options(visible_idxs=[1])
utils.set_random_seed(0)

from ray.tune.registry import register_env
register_env("eth2-v1", lambda config: Eth2v1(config))

if __name__=='__main__':
    dataset = Dataset(start_time='2021-08-01 00:00:00')
    print('Check env:')
    ray.rllib.utils.check_env(Eth2v1(config=dict(txs=dataset.txs, allocate=GroupAllocateStrategy(6,7), n_blocks=10, tx_rate=500, tx_per_block=200, block_interval=15)))
    
