# Import the RL algorithm (Algorithm) we would like to use.
import ray
from ray.rllib.algorithms.dqn import DQN
from env import Eth2v1
from arrl.dataset import Dataset
from strategy import GroupAllocateStrategy
import utils
import tqdm
utils.tfutil.set_gpu_options(visible_idxs=[1])
utils.set_random_seed(0)

from ray.tune.registry import register_env
register_env("eth2-v1", lambda config: Eth2v1(config))

if __name__=='__main__':
    # Configure the algorithm.
    config = {
        "horizon": 1,
        "soft_horizon": True,
        # Environment
        "env": "eth2-v1",
        "env_config":{
            "txs":"Dataset(start_time='2021-08-01 00:00:00').txs", 
            "k":3,
            "g":7,
            "n_blocks":10,
            "tx_rate":100,
            "tx_per_block":200,
            "block_interval":15
        },
        # Use environment workers (aka "rollout workers") that parallelly
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
        "evaluation_num_workers": 0,
        "evaluation_config": {
            "evaluation_interval": None,
            "render_env": False
        },
        "train_batch_size": 1024,
        "gamma": 0.99,
        "lr": 0.01
    }

    # Create our RLlib Trainer.
    algo = PPO(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    pbar = tqdm.trange(100)
    for i in pbar:
        print(algo.train())
    
    print("Evaluation:")
    env = Eth2v1(config=config['env_config'])
    obs = env.reset()
    done = 0
    while not done:
        action = algo.compute_single_action(
            observation=obs,
            explore=False
        )
        obs, reward, done, _ = env.step(action)
    print(env.info())