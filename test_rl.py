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

    # Configure the algorithm.
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "eth2-v1",
        "env_config":{
            "txs":dataset.txs, 
            "allocate":GroupAllocateStrategy(6,7),
            "n_blocks":10,
            "tx_rate":500,
            "tx_per_block":200,
            "block_interval":15
        },
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "tf2",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": False,
        },
    }

    # Create our RLlib Trainer.
    algo = PPO(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(3):
        print(algo.train())
    
    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
    algo.evaluate()