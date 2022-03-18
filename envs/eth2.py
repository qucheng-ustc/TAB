import gym
import pandas as pd
import numpy as np

class Eth2(gym.Env):
    def __init__(self):
        print("Eth2 initialized")

    def init(self, txs, allocate, n_samples=None, **kwargs):
        if n_samples is None:
            self.txs = txs
        else:
            self.txs = txs.sample(n_samples)
        self.tx_rate = 1000
        self.tx_per_block = 200
        self.block_interval = 15

        self.allocate = allocate
        self.k = allocate.k
        self.n_shards = 1<<self.k

        self.config(**kwargs)

        self.reset()
       
    def config(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
            if key == 'allocate':
                self.k = self.allocate.k
                self.n_shards = 1<<self.k

    def step(self, action=None):
        print("Eth2 step", action)
        allocate = self.allocate
        n_shards = self.n_shards
        ptx = self.ptx
        self.ptx += 1
        tx = self.txs.iloc[ptx]
        s_from = allocate.allocate(tx['from'])
        s_to = allocate.allocate(tx['to'])
        tx = {'id':self.tx_count, 's_from':s_from, 's_to':s_to, 'time':self.simulate_time}
        self.shard_tx_pool[tx['s_from']].append(tx)
        self.tx_count += 1
        if self.tx_count % self.tx_rate == 0:
            self.simulate_time += 1
            for i in range(n_shards):
                self.shard_queue_length[i].append(len(self.shard_tx_pool[i]))
        if self.tx_count % (self.tx_rate * self.block_interval) == 0:
            for i in range(n_shards):
                tx_pool = self.shard_tx_pool[i]
                n_tx = 0
                n_from_tx = 0
                n_to_tx = 0
                for n in range(self.tx_per_block):
                    if len(tx_pool)==0:
                        break
                    tx = tx_pool.pop(0)
                    if tx['s_from'] == i:
                        n_from_tx += 1
                        if tx['s_from'] != tx['s_to']:
                            self.n_shard_cross_tx[i] += 1
                            self.n_cross_tx += 1
                            self.shard_forward_tx_pool[tx['s_to']].append(tx)
                        else:
                            self.n_shard_inner_tx[i] += 1
                            self.n_inner_tx += 1
                    else:
                        n_to_tx += 1
                    n_tx += 1
                block = {'id':self.block_count, 'n_tx':n_tx, 'n_from_tx':n_from_tx, 'n_to_tx':n_to_tx}
                self.block_count += 1
                self.shard_blocks[i].append(block)
            for i in range(n_shards):
                forward_tx_pool = self.shard_forward_tx_pool[i]
                while len(forward_tx_pool)>0:
                    tx = forward_tx_pool.pop(0)
                    self.shard_tx_pool[i].insert(0,tx)

        observation = [len(self.shard_tx_pool[i]) for i in range(n_shards)]
        reward = 0
        done = 0
        info = None

        return observation, reward, done, info

    def info(self):
        n_block_tx = 0
        n_from_tx = 0
        n_to_tx = 0
        n_shards = self.n_shards
        n_shard_tx = [0 for i in range(n_shards)]
        for i in range(n_shards):
            for block in self.shard_blocks[i]:
                n_shard_tx[i] += block['n_tx']
                n_block_tx += block['n_tx']
                n_from_tx += block['n_from_tx']
                n_to_tx += block['n_to_tx']
        result = {
            'n_shards': self.n_shards,
            'n_tx': self.n_inner_tx + self.n_cross_tx,
            'n_inner_tx': self.n_inner_tx,
            'n_cross_tx': self.n_cross_tx,
            'n_shard_cross_tx': self.n_shard_cross_tx,
            'n_shard_inner_tx': self.n_shard_inner_tx,
            'cross_rate': self.n_cross_tx/(self.n_inner_tx+self.n_cross_tx),
            'simulate_time': self.simulate_time,
            'n_block_tx': n_block_tx,
            'n_from_tx': n_from_tx,
            'n_to_tx': n_to_tx,
            'throughput': n_block_tx/self.simulate_time,
            'actual_throughput': n_from_tx/self.simulate_time,
            'ideal_throughput': self.n_shards*self.tx_per_block/self.block_interval,
            'shard_queue_length': self.shard_queue_length
        }
        return result

    def reset(self):
        print("Eth2 reset")
        n_shards = self.n_shards
        self.ptx = 0
        self.n_shard_cross_tx = [0 for i in range(n_shards)]
        self.n_shard_inner_tx = [0 for i in range(n_shards)]
        self.shard_tx_pool = [[] for i in range(n_shards)]
        self.shard_forward_tx_pool = [[] for i in range(n_shards)]
        self.shard_blocks = [[] for i in range(n_shards)]
        self.shard_queue_length = [[] for i in range(n_shards)]

        self.tx_count = 0
        self.block_count = 0
        self.simulate_time = 0

        self.n_cross_tx = 0
        self.n_inner_tx = 0

class Eth2v1(Eth2):
    # config: n_blocks: number of blocks per step
    def step(self, action=None):
        print("Eth2 v1 step", action)
        # one step contains several blocks
        n_blocks = self.n_blocks
        n_shards = self.n_shards
        allocate = self.allocate
        tx_per_block = self.tx_per_block
        block_interval = self.block_interval
        tx_rate = self.tx_rate
        tx_count = block_interval * tx_rate

        epoch_time = n_blocks * block_interval
        self.simulate_time += epoch_time

        for b in range(n_blocks):
            txs = self.txs.iloc[self.ptx:self.ptx+tx_count]
            self.ptx += tx_count
            self.tx_count += tx_count
            for i, tx in txs.iterrows():
                s_from = allocate.allocate(tx['from'])
                s_to = allocate.allocate(tx['to'])
                tx = {'s_from':s_from, 's_to':s_to}
                self.shard_tx_pool[tx['s_from']].append(tx)
                self.tx_count += 1
            for i in range(n_shards):
                self.shard_queue_length[i].append(len(self.shard_tx_pool[i]))
            for i in range(n_shards):
                tx_pool = self.shard_tx_pool[i]
                n_tx = 0
                n_from_tx = 0
                n_to_tx = 0
                for n in range(self.tx_per_block):
                    if len(tx_pool)==0:
                        break
                    tx = tx_pool.pop(0)
                    if tx['s_from'] == i:
                        n_from_tx += 1
                        if tx['s_from'] != tx['s_to']:
                            self.n_shard_cross_tx[i] += 1
                            self.n_cross_tx += 1
                            self.shard_forward_tx_pool[tx['s_to']].append(tx)
                        else:
                            self.n_shard_inner_tx[i] += 1
                            self.n_inner_tx += 1
                    else:
                        n_to_tx += 1
                    n_tx += 1
                block = {'id':self.block_count, 'n_tx':n_tx, 'n_from_tx':n_from_tx, 'n_to_tx':n_to_tx}
                self.block_count += 1
                self.shard_blocks[i].append(block)
            for i in range(n_shards):
                forward_tx_pool = self.shard_forward_tx_pool[i]
                while len(forward_tx_pool)>0:
                    tx = forward_tx_pool.pop(0)
                    self.shard_tx_pool[i].insert(0,tx)

        return self.shard_blocks, self.shard_tx_pool, self.n_shard_cross_tx, self.n_shard_inner_tx

class Eth2v2(Eth2):
    def step(self, action=None):
        print("Eth2 v2 step", action)
        # one step contains several blocks
        n_blocks = self.n_blocks
        n_shards = self.n_shards
        allocate = self.allocate
        tx_per_block = self.tx_per_block
        block_interval = self.block_interval
        tx_rate = self.tx_rate
        tx_count = block_interval * tx_rate

        epoch_time = n_blocks * block_interval
        self.simulate_time += epoch_time

        epoch_tx_count = tx_count * n_blocks
        txs = self.txs.iloc[self.ptx : self.ptx+epoch_tx_count]
        self.ptx += epoch_tx_count

        s_from = txs['from'].apply(allocate.allocate) # from shard index
        s_to = txs['to'].apply(allocate.allocate) # to shard index

        self.n_cross_tx += len(s_from[s_from!=s_to])
        self.n_inner_tx += len(s_from[s_from==s_to])

        sgroup = s_to.groupby(s_from)
        stx_pool = self.stx_pool
        stx_forward = self.stx_forward
        for g, group in sgroup:
            stx_pool[g].extend(group)
        
        for b in range(n_blocks):
            for s in range(n_shards):
                tx_pool = stx_pool[s]
                tx_forward = stx_forward[s]
                n_left = tx_per_block
                if len(tx_forward)>0:
                    if n_left < len(tx_forward):
                        block_txs = tx_forward[:tx_per_block]
                        stx_forward[s] = tx_forward[tx_per_block:]
                        n_left = 0
                    else:
                        block_txs = tx_forward
                        stx_forward[s] = []
                        n_left -= len(block_txs)
                else:
                    block_txs = []
                if n_left>0:
                    if n_left < len(tx_pool):
                        block_txs.extend(tx_pool[:n_left])
                        stx_pool[s] = tx_pool[n_left:]
                    else:
                        block_txs.extend(tx_pool)
                        stx_pool[s] = []
                self.sblock[s].append(block_txs)
            for s in range(n_shards):
                block = self.sblock[s][-1]
                for tx in block:
                    if abs(tx)!=s:
                        stx_forward[tx].append(-tx)
        return self.stx_pool, self.sblock

    def reset(self):
        self.ptx = 0
        self.n_cross_tx = 0
        self.n_inner_tx = 0
        self.simulate_time = 0

        self.stx_pool = [[] for i in range(self.n_shards)]
        self.stx_forward = [[] for i in range(self.n_shards)]
        self.sblock = [[] for i in range(self.n_shards)]

    def info(self):
        n_block_tx = 0
        n_block_out_tx = 0
        n_block_forward_tx = 0
        n_block_inner_tx = 0
        for s in range(self.n_shards):
            blocks = self.sblock[s]
            for block in blocks:
                for tx in block:
                    if abs(tx)!=s:
                        n_block_out_tx += 1
                    elif tx>0:
                        n_block_inner_tx += 1
                    else:
                        n_block_forward_tx += 1
                    n_block_tx += 1

        result = dict(
            n_shards = self.n_shards,
            n_tx = self.ptx,
            n_inner_tx = self.n_inner_tx,
            n_cross_tx = self.n_cross_tx,
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            simulate_time = self.simulate_time,
            throughput = n_block_tx/self.simulate_time,
            actual_throughput = (n_block_inner_tx+n_block_forward_tx)/self.simulate_time,
            ideal_throughput = self.tx_per_block*self.n_shards/self.block_interval,
            tx_pool_length = [len(pool) for pool in self.stx_pool],
            tx_forward_length = [len(forward) for forward in self.stx_forward]
        )
        return result

from collections import deque

class Eth2v3(Eth2v2):
    #params: 
    #  n_blocks: number of blocks per step
    def step(self, action=None):
        print('Eth2 v3 step', action)
        # one step contains several blocks
        n_blocks = self.n_blocks
        n_shards = self.n_shards
        allocate = self.allocate
        allocate.apply(action)
        tx_per_block = self.tx_per_block
        block_interval = self.block_interval
        tx_rate = self.tx_rate
        tx_count = block_interval * tx_rate

        epoch_time = n_blocks * block_interval
        self.simulate_time += epoch_time

        epoch_tx_count = tx_count * n_blocks
        txs = self.txs.iloc[self.ptx : self.ptx+epoch_tx_count]
        self.ptx += epoch_tx_count

        s_from = txs['from'].apply(allocate.allocate) # from shard index
        s_to = txs['to'].apply(allocate.allocate) # to shard index

        n_cross = len(s_from[s_from!=s_to])
        self.n_cross_tx += n_cross
        self.n_inner_tx += len(s_from)-n_cross

        sgroup = s_to.groupby(s_from)
        stx_pool = self.stx_pool
        stx_forward = self.stx_forward
        for g, group in sgroup:
            stx_pool[g].extend(group)
        
        for b in range(n_blocks):
            for s in range(n_shards):
                tx_pool = stx_pool[s]
                tx_forward = stx_forward[s]
                block_txs = deque()
                n_forward = min(len(tx_forward), tx_per_block)
                block_txs.extend(tx_forward.popleft() for _ in range(n_forward))
                n_pool = min(len(tx_pool), tx_per_block-len(block_txs))
                block_txs.extend(tx_pool.popleft() for _ in range(n_pool))
                self.sblock[s].append(block_txs)
            for s in range(n_shards):
                block = self.sblock[s][-1]
                for tx in block:
                    if abs(tx)!=s:
                        stx_forward[tx].append(-tx)
        return self.stx_pool, self.sblock

    def reset(self):
        self.ptx = 0
        self.n_cross_tx = 0
        self.n_inner_tx = 0
        self.simulate_time = 0

        self.stx_pool = [deque() for i in range(self.n_shards)]
        self.stx_forward = [deque() for i in range(self.n_shards)]
        self.sblock = [deque() for i in range(self.n_shards)]
