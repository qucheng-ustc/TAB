import numpy as np
import gym
from gym.envs.registration import register
from collections import OrderedDict
from gym.spaces import MultiDiscrete, Box, Dict
from env.eth2 import Eth2v1Simulator, Eth2v2Simulator

def min_max_scale(a, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(a)
    if max_val is None:
        max_val = np.max(a)
    if min_val == max_val:
        return np.zeros_like(a)
    return (a-min_val)/(max_val-min_val)

class Eth2v1(gym.Env):
    #params: 
    #  n_blocks: number of blocks per step
    def __init__(self, config={}):
        self.tx_rate = config.get("tx_rate",1000)
        self.tx_per_block = config.get("tx_per_block",200)
        self.block_interval = config.get("block_interval",15)
        self.n_blocks = config.get("n_blocks",10)
        self.addr_len = config.get("addr_len", 16)
        if "txs" in config:
            self.txs = config.get("txs")
        else:
            from arrl.dataset import RandomDataset
            self.txs = RandomDataset(size=10000000).txs
        if isinstance(self.txs, str):
            from arrl.dataset import Dataset, RandomDataset
            self.txs = eval(self.txs)
        if "allocate" in config:
            self.allocate = config.get("allocate")
            self.k = self.allocate.k
            self.g = self.allocate.g
        else:
            from strategy import GroupAllocateStrategy
            self.k = config.get("k",6)
            self.g = config.get("g",7)
            self.allocate = GroupAllocateStrategy(self.k,self.g)
        self.n_shards = 1<<self.k
        self.n_accounts = 1<<self.g
        if 'simulator' in config:
            self.simulator = config.get("simulator")
        else:
            self.simulator = Eth2v1Simulator(txs=self.txs, allocate=self.allocate, n_shards=self.n_shards, tx_rate=self.tx_rate,
                    tx_per_block=self.tx_per_block, block_interval=self.block_interval, n_blocks=self.n_blocks)
        self.action_space = MultiDiscrete([self.n_shards]*self.n_accounts)
        self.observation_space = Dict({
            'adj_matrix':Box(low=0.0,high=1.0,shape=(self.n_accounts,self.n_accounts)), # adjacency matrix (weighted by number of tx)
            'degree':Box(low=0.0,high=1.0,shape=(self.n_accounts,)), # account degree (number of tx)
            'feature':Box(low=0.0,high=1.0,shape=(self.n_accounts,)), # account feature (total gas)
            'partition':Box(low=0.0,high=1.0,shape=(self.n_accounts,self.n_shards))}) # last partition

    def reset(self):
        self.done = self.simulator.reset()
        self.partition_table = np.eye(self.n_shards)[self.allocate.group_table]
        self.adj_matrix = np.eye(self.n_accounts)
        self.degree = np.ones(shape=self.n_accounts)
        self.feature = np.zeros(shape=self.n_accounts)
        self.reward = 0
        self.target_throughput = self.tx_per_block*self.n_shards/self.block_interval
        return self.observation()

    def step(self, action):
        self.done = self.simulator.step(action)
        self.partition_table = np.eye(self.n_shards)[self.allocate.group_table]
        self.adj_matrix = np.zeros(shape=(self.n_accounts,self.n_accounts))
        self.degree = np.zeros(shape=self.n_accounts)
        self.feature = np.zeros(shape=self.n_accounts)
        n_block_out_tx, n_block_inner_tx, n_block_forward_tx = 0, 0, 0
        for shard, blocks in enumerate(self.simulator.sblock):
            n_blocks = min(len(blocks),self.n_blocks)
            for block in blocks[-n_blocks:]:
                for from_addr, to_addr, gas, from_shard, to_shard in block:
                    from_group, to_group = self.allocate.group(from_addr), self.allocate.group(to_addr)
                    self.adj_matrix[from_group, to_group] += 1
                    self.adj_matrix[to_group, from_group] += 1
                    self.degree[from_group] += 1
                    self.degree[to_group] += 1
                    self.feature[from_group] += gas
                    if to_shard!=shard:
                        n_block_out_tx += 1
                    elif from_shard==shard:
                        n_block_inner_tx += 1
                    else:
                        n_block_forward_tx += 1
        actual_throughput = (n_block_inner_tx+n_block_forward_tx)/self.simulator.simulate_time
        self.reward = actual_throughput/self.target_throughput
        return self.observation(), self.reward, self.done, {}

    def observation(self):
        return OrderedDict(adj_matrix=min_max_scale(self.adj_matrix, min_val=0), degree=min_max_scale(self.degree, min_val=0), feature=min_max_scale(self.feature, min_val=0), partition=self.partition_table)

    def info(self):
        return self.simulator.info()

class Eth2v2(Eth2v1):
    def __init__(self, config={}):
        if 'simulator' not in config:
            config['simulator'] = None
        super().__init__(config)
        if self.simulator is None:
            self.simulator = Eth2v2Simulator(txs=self.txs, allocate=self.allocate, n_shards=self.n_shards, tx_rate=self.tx_rate,
                    tx_per_block=self.tx_per_block, block_interval=self.block_interval, n_blocks=self.n_blocks)

register(
        id='eth2-v1',
        entry_point='env.gym:Eth2v1'
)

register(
        id='eth2-v2',
        entry_point='env.gym:Eth2v2'
)
