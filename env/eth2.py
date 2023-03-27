import gym
import pandas as pd
import numpy as np
import itertools
from collections import deque, OrderedDict
from gym.spaces import MultiDiscrete, Box, Dict

class Eth2v1Simulator:
    def __init__(self, client, allocate, n_shards, tx_per_block=200, block_interval=15, n_blocks=10):
        self.tx_per_block = tx_per_block
        self.block_interval = block_interval
        self.n_blocks = n_blocks
        self.allocate = allocate
        self.n_shards = n_shards

        self.client = client
        self.txs = client.txs
        self.tx_rate = client.tx_rate

        self.tx_count = block_interval * self.tx_rate
        self.epoch_time = n_blocks * block_interval
        self.epoch_tx_count = self.tx_count * n_blocks

        self.max_epochs = len(self.txs)//self.epoch_tx_count

    def reset(self, ptx=0):
        self.allocate.reset()
        self.client.reset(ptx=ptx)
        self.n_cross_tx = 0
        self.n_inner_tx = 0
        self.simulate_time = 0

        self.stx_pool = [deque() for _ in range(self.n_shards)]
        self.stx_forward = [deque() for _ in range(self.n_shards)]
        self.sblock = [[] for _ in range(self.n_shards)]

        return self.client.done(time_interval=self.epoch_time)

    def step(self, action):
        # one step contains n_blocks blocks
        self.simulate_time += self.epoch_time
        self.allocate.apply(action) # apply allocate action before txs arrives
        txs = self.client.next(time_interval=self.epoch_time).copy()
        self.epoch_txs = txs
        
        txs['from_shard'] = txs['from'].map(self.allocate.allocate) # from shard index
        txs['to_shard'] = txs['to'].map(self.allocate.allocate) # to shard index

        counts = (txs['from_shard'] == txs['to_shard']).value_counts()
        self.n_inner_tx += counts.get(True, default=0)
        self.n_cross_tx += counts.get(False, default=0)

        for from_shard, tx in txs.groupby('from_shard'):
            #txs store in tuple: (from_addr, to_addr, gas, from_shard, to_shard)
            self.stx_pool[from_shard].extend(tx.itertuples(index=False, name=None))

        for _ in range(self.n_blocks):
            for shard, (tx_pool, tx_forward) in enumerate(zip(self.stx_pool, self.stx_forward)):
                n_forward = min(len(tx_forward), self.tx_per_block)
                n_pool = min(len(tx_pool), self.tx_per_block-n_forward)
                block_txs = [tx_forward.popleft() for _ in range(n_forward)]+[tx_pool.popleft() for _ in range(n_pool)]
                self.sblock[shard].append(block_txs)
            for shard, blocks in enumerate(self.sblock):
                for tx in blocks[-1]:
                    to_shard = tx[4]
                    if to_shard!=shard:
                        self.stx_forward[to_shard].append(tx)

        return self.client.done(time_interval=self.epoch_time)

    def info(self):
        n_tx = self.client.n_tx()
        n_block = 0
        n_block_tx = 0
        n_block_out_tx = 0
        n_block_forward_tx = 0
        n_block_inner_tx = 0
        tx_wasted = [0 for _ in range(self.n_shards)]
        for shard, blocks in enumerate(self.sblock):
            n_block += len(blocks)
            for block in blocks:
                n_block_tx += len(block)
                for _, _, _, from_shard, to_shard in block:
                    if to_shard!=shard:
                        n_block_out_tx += 1
                    elif from_shard==shard:
                        n_block_inner_tx += 1
                    else:
                        n_block_forward_tx += 1
                tx_wasted[shard] += self.tx_per_block - len(block)
        n_wasted = sum(tx_wasted)

        result = dict(
            n_shards = self.n_shards,
            blocks_per_epoch = self.n_blocks,
            tx_rate = self.tx_rate,
            tx_per_block = self.tx_per_block,
            block_interval = self.block_interval,
            simulate_time = self.simulate_time,
            n_block = n_block,
            target_n_block = self.n_shards*self.simulate_time/self.block_interval,
            n_tx = n_tx,
            n_inner_tx = self.n_inner_tx,
            n_cross_tx = self.n_cross_tx,
            prop_cross_tx = float(self.n_cross_tx) / n_tx if n_tx>0 else 0,
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            throughput = n_block_tx/self.simulate_time if self.simulate_time>0 else 0,
            actual_throughput = (n_block_inner_tx+n_block_forward_tx)/self.simulate_time if self.simulate_time>0 else 0,
            target_throughput = self.tx_per_block*self.n_shards/self.block_interval,
            tx_pool_length = [len(pool) for pool in self.stx_pool],
            tx_forward_length = [len(forward) for forward in self.stx_forward],
            n_wasted = n_wasted,
            tx_wasted = tx_wasted,
            prop_wasted = float(n_wasted) / (n_block * self.tx_per_block)
        )
        result['prop_throughput'] = float(result['actual_throughput'])/result['target_throughput']
        return result

class Eth2v2Simulator(Eth2v1Simulator):
    def step(self, action):
        self.allocate.apply(action) # apply allocate action before txs arrives
        txs = self.client.next(time_interval=self.epoch_time).copy() # prepare new transactions
        txs['from_shard'] = txs['from'].map(self.allocate.allocate) # from shard index
        txs['to_shard'] = txs['to'].map(self.allocate.allocate) # to shard index

        counts = (txs['from_shard'] == txs['to_shard']).value_counts()
        self.n_inner_tx += counts.get(True, default=0)
        self.n_cross_tx += counts.get(False, default=0)

        # re-allocate tx pool
        stx_pool = [deque() for i in range(self.n_shards)]
        stx_forward = [deque() for i in range(self.n_shards)]
        for shard, (tx_pool, tx_forward) in enumerate(zip(self.stx_pool, self.stx_forward)):
            for from_addr, to_addr, gas, _, _ in tx_pool:
                from_shard = self.allocate.allocate(from_addr)
                to_shard = self.allocate.allocate(to_addr)
                stx_pool[from_shard].append((from_addr, to_addr, gas, from_shard, to_shard))
            for from_addr, to_addr, gas, _, _ in tx_forward:
                from_shard = self.allocate.allocate(from_addr)
                to_shard = self.allocate.allocate(to_addr)
                stx_forward[to_shard].append((from_addr, to_addr, gas, from_shard, to_shard))
        self.stx_pool = stx_pool
        self.stx_forward = stx_forward

        # start simulation
        for slot_ptx in range(0, len(txs), self.tx_count):
            # in each time slot, tx_rate new transactions arrived
            slot_txs = txs.iloc[slot_ptx: min(slot_ptx+self.tx_count, len(txs))]
            for from_shard, tx in slot_txs.groupby('from_shard'):
                #txs store in tuple: (from_addr, to_addr, gas, from_shard, to_shard)
                stx_pool[from_shard].extend(tx.itertuples(index=False, name=None))
            # each shard produce one block
            for shard, (tx_pool, tx_forward) in enumerate(zip(stx_pool, stx_forward)):
                n_forward = min(len(tx_forward), self.tx_per_block)
                n_pool = min(len(tx_pool), self.tx_per_block-n_forward)
                block_txs = [tx_forward.popleft() for _ in range(n_forward)]+[tx_pool.popleft() for _ in range(n_pool)]
                self.sblock[shard].append(block_txs)
            # cross shard tx forward to target shard in next timeslot
            for shard, blocks in enumerate(self.sblock):
                for tx in blocks[-1]:
                    to_shard = tx[4]
                    if to_shard!=shard:
                        stx_forward[to_shard].append(tx)
            self.simulate_time += self.block_interval
        self.next_txs = self.client.next(time_interval=self.epoch_time)
        return self.client.done()

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

