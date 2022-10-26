import gym
import pandas as pd
import numpy as np
import itertools

class Eth2(gym.Env):
    def __init__(self):
        self.tx_rate = 1000
        self.tx_per_block = 200
        self.block_interval = 15

    def init(self, txs, allocate, **kwargs):
        self.txs = txs
        self.allocate = allocate
        self.k = allocate.k
        self.n_shards = 1<<self.k

        self.config(**kwargs)

        self.reset()

        return self
       
    def config(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
            if key == 'allocate':
                self.k = self.allocate.k
                self.n_shards = 1<<self.k
        return self

    def step(self, action=None):
        print("Eth2 step", action)
        allocate = self.allocate
        n_shards = self.n_shards
        ptx = self.ptx
        self.ptx += 1
        tx = self.txs.iloc[ptx]
        s_from = allocate.allocate(tx['from_addr'])
        s_to = allocate.allocate(tx['to_addr'])
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
                s_from = allocate.allocate(tx['from_addr'])
                s_to = allocate.allocate(tx['to_addr'])
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
    #params: 
    #  n_blocks: number of blocks per step
    def __init__(self):
        super().__init__()
        self.n_blocks = 10

    def step(self, action=None):
        #print("Eth2 v2 step", action)
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

        s_from = txs['from_addr'].apply(allocate.allocate) # from shard index
        s_to = txs['to_addr'].apply(allocate.allocate) # to shard index

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
        n_block = 0
        n_block_tx = 0
        n_block_out_tx = 0
        n_block_forward_tx = 0
        n_block_inner_tx = 0
        tx_wasted = [0 for i in range(self.n_shards)]
        for s in range(self.n_shards):
            blocks = self.sblock[s]
            n_block += len(blocks)
            for block in blocks:
                n_block_tx += len(block)
                for tx in block:
                    if abs(tx)!=s:
                        n_block_out_tx += 1
                    elif tx>0:
                        n_block_inner_tx += 1
                    else:
                        n_block_forward_tx += 1
                tx_wasted[s] += self.tx_per_block - len(block)
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
            n_tx = self.ptx,
            n_inner_tx = self.n_inner_tx,
            n_cross_tx = self.n_cross_tx,
            prop_cross_tx = float(self.n_cross_tx) / self.ptx,
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            throughput = n_block_tx/self.simulate_time,
            actual_throughput = (n_block_inner_tx+n_block_forward_tx)/self.simulate_time,
            target_throughput = self.tx_per_block*self.n_shards/self.block_interval,
            tx_pool_length = [len(pool) for pool in self.stx_pool],
            tx_forward_length = [len(forward) for forward in self.stx_forward],
            n_wasted = n_wasted,
            tx_wasted = tx_wasted,
            prop_wasted = float(n_wasted) / (n_block * self.tx_per_block)
        )
        result['prop_throughput'] = float(result['actual_throughput'])/result['target_throughput']
        return result

from collections import deque

class Eth2v301(Eth2v2):
    #params: 
    #  n_blocks: number of blocks per step
    def step(self, action=None):
        # print('Eth2 v3 step', action)
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

        s_from = txs['from_addr'].apply(allocate.allocate) # from shard index
        s_to = txs['to_addr'].apply(allocate.allocate) # to shard index

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

        return self.stx_pool, self.sblock

class Eth2v302(Eth2v2):
    #params: 
    #  n_blocks: number of blocks per step
    def step(self, action=None):
        # print('Eth2 v3 step', action)
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
        txs = self.txs.iloc[self.ptx:min(self.ptx+epoch_tx_count, len(self.txs))]
        self.ptx += len(txs)
        done = self.ptx >= len(self.txs)

        s_from = txs['from_addr'].map(allocate.allocate) # from shard index
        s_to = txs['to_addr'].map(allocate.allocate) # to shard index

        counts = (s_from == s_to).value_counts()
        self.n_inner_tx += counts.get(True, default=0)
        self.n_cross_tx += counts.get(False, default=0)

        sgroup = s_to.groupby(s_from)
        stx_pool = self.stx_pool
        stx_forward = self.stx_forward
        for g, group in sgroup:
            stx_pool[g].extend(group)
        
        for _ in range(n_blocks):
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

        observation = [self.stx_pool, self.stx_forward, self.sblock]
        reward = 0
        return observation, reward, done, None

    def reset(self):
        self.ptx = 0
        self.n_cross_tx = 0
        self.n_inner_tx = 0
        self.simulate_time = 0

        self.stx_pool = [deque() for _ in range(self.n_shards)]
        self.stx_forward = [deque() for _ in range(self.n_shards)]
        self.sblock = [deque() for _ in range(self.n_shards)]

        observation = [self.stx_pool, self.stx_forward, self.sblock]

        return observation, None, 0, None

class Eth2v3(Eth2v2):
    #params: 
    #  n_blocks: number of blocks per step
    def reset(self):
        self.ptx = 0
        self.n_cross_tx = 0
        self.n_inner_tx = 0
        self.simulate_time = 0

        self.stx_pool = [deque() for _ in range(self.n_shards)]
        self.stx_forward = [deque() for _ in range(self.n_shards)]
        self.sblock = [deque() for _ in range(self.n_shards)]

        observation = [self.stx_pool, self.stx_forward, self.sblock]

        return observation, None, 0, None

    def step(self, action=None):
        # print('Eth2 v3 step', action)
        # one step contains n_blocks blocks
        n_blocks = self.n_blocks
        allocate = self.allocate
        allocate.apply(action)
        tx_per_block = self.tx_per_block
        block_interval = self.block_interval
        tx_rate = self.tx_rate
        tx_count = block_interval * tx_rate

        epoch_time = n_blocks * block_interval
        self.simulate_time += epoch_time

        epoch_tx_count = tx_count * n_blocks
        txs = self.txs.iloc[self.ptx:min(self.ptx+epoch_tx_count, len(self.txs))].copy()
        self.ptx += len(txs)
        done = self.ptx >= len(self.txs)

        txs['from_shard'] = txs['from_addr'].map(allocate.allocate) # from shard index
        txs['to_shard'] = txs['to_addr'].map(allocate.allocate) # to shard index

        counts = (txs['from_shard'] == txs['to_shard']).value_counts()
        self.n_inner_tx += counts.get(True, default=0)
        self.n_cross_tx += counts.get(False, default=0)

        stx_pool = self.stx_pool
        stx_forward = self.stx_forward
        for from_shard, tx in txs.groupby('from_shard'):
            #txs store in tuple: (from_addr, to_addr, from_shard, to_shard)
            stx_pool[from_shard].extend(tx.itertuples(index=False, name=None))

        for _ in range(n_blocks):
            for shard, (tx_pool, tx_forward) in enumerate(zip(stx_pool, stx_forward)):
                n_forward = min(len(tx_forward), tx_per_block)
                n_pool = min(len(tx_pool), tx_per_block-n_forward)
                block_txs = [tx_forward.popleft() for _ in range(n_forward)]+[tx_pool.popleft() for _ in range(n_pool)]
                self.sblock[shard].append(block_txs)
            for shard, blocks in enumerate(self.sblock):
                for tx in blocks[-1]:
                    to_shard = tx[3]
                    if to_shard!=shard:
                        stx_forward[to_shard].append(tx)

        observation = [self.stx_pool, self.stx_forward, self.sblock]
        reward = 0
        return observation, reward, done, None

    def info(self):
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
                for _, _, from_shard, to_shard in block:
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
            n_tx = self.ptx,
            n_inner_tx = self.n_inner_tx,
            n_cross_tx = self.n_cross_tx,
            prop_cross_tx = float(self.n_cross_tx) / self.ptx,
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            throughput = n_block_tx/self.simulate_time,
            actual_throughput = (n_block_inner_tx+n_block_forward_tx)/self.simulate_time,
            target_throughput = self.tx_per_block*self.n_shards/self.block_interval,
            tx_pool_length = [len(pool) for pool in self.stx_pool],
            tx_forward_length = [len(forward) for forward in self.stx_forward],
            n_wasted = n_wasted,
            tx_wasted = tx_wasted,
            prop_wasted = float(n_wasted) / (n_block * self.tx_per_block)
        )
        result['prop_throughput'] = float(result['actual_throughput'])/result['target_throughput']
        return result

class TxBuffView:
    def __init__(self, buff, start, length):
        self.buff_size = len(buff)
        self.buff = buff
        self.start = start
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else self.length
            if start < 0:
                start += self.length
            if stop < 0:
                stop += self.length
            assert(0<=start and start<=stop and stop<=self.length)
            return TxBuffView(self.buff, (start+self.start)%self.buff_size, stop-start)
        if i < 0:
            i += self.length
        assert(i<self.length)
        return self.buff[(self.start+i)%self.buff_size]
    
    def __iter__(self):
        for i in range(self.start, self.start+self.length):
            yield self.buff[i%self.buff_size]
    
    def __str__(self):
        return str(list(x for x in self))
    
    def __repr__(self):
        return f'TxBuffView(buff_size={self.buff_size},start={self.start},length={self.length}):'+str(self)

class TxBuff(TxBuffView):
    def __init__(self, buff_size, dtype=np.int32, allow_growth=True):
        self.dtype = dtype
        self.allow_growth = allow_growth
        super().__init__(buff=np.zeros(shape=buff_size, dtype=dtype), start=0, length=0)

    def grow(self):
        self.buff.resize(self.buff_size + self.buff_size)
        self.buff[self.buff_size:] = self.buff[:self.buff_size]
        self.buff_size += self.buff_size

    def extend(self, txs):
        n = len(txs)
        while self.length + n > self.buff_size:
            if not self.allow_growth:
                raise OverflowError
            self.grow()
        end = (self.start + self.length) % self.buff_size
        if end + n <= self.buff_size:
            self.buff[end:end+n] = txs
        else:
            front = self.buff_size-end
            self.buff[end:] = txs[:front]
            rest = n - front
            self.buff[:rest] = txs[front:]
        self.length += n

    def append(self, tx):
        if self.length >= self.buff_size:
            if not self.allow_growth:
                raise OverflowError
            self.grow()
        self.buff[(self.start + self.length) % self.buff_size] = tx
        self.length += 1
    
    def popnleft(self, n):
        assert(self.length>=n)
        ret = self[:n]
        self.start = (self.start + n) % self.buff_size
        self.length -= n
        return ret

#from eth2v4 import TxBuffView, TxBuff

class Eth2v4(gym.Env):
    def __init__(self):
        #print("Eth2v4 initialized")
        pass

    def init(self, txs, allocate, n_samples=None, tx_rate=1000, tx_per_block=200, block_interval=15, n_blocks=10, pool_size=10000000):
        if n_samples is None:
            self.txs = txs
        else:
            self.txs = txs.sample(n_samples)
        self.tx_rate = tx_rate
        self.tx_per_block = tx_per_block
        self.block_interval = block_interval

        self.allocate = allocate
        self.k = allocate.k
        self.n_shards = 1<<self.k

        self.n_blocks = n_blocks
        self.pool_size = pool_size

        self.reset()

    def reset(self):
        self.ptx = 0
        self.n_cross_tx = 0
        self.n_inner_tx = 0
        self.simulate_time = 0

        self.stx_pool = [TxBuff(self.pool_size) for i in range(self.n_shards)]
        self.stx_forward = [TxBuff(self.pool_size) for i in range(self.n_shards)]
        self.sblock = [deque() for i in range(self.n_shards)]

        observation = [self.stx_pool, self.stx_forward, self.sblock]

        return observation, None, 0, None

    def step(self, action=None):
        # print('Eth2 v4 step', action)
        # one step contains self.n_blocks blocks
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

        s_from = txs['from_addr'].apply(allocate.allocate) # from shard index
        s_to = txs['to_addr'].apply(allocate.allocate) # to shard index

        n_cross = len(s_from[s_from!=s_to])
        self.n_cross_tx += n_cross
        self.n_inner_tx += len(s_from)-n_cross

        sgroup = s_to.groupby(s_from)
        stx_pool = self.stx_pool
        stx_forward = self.stx_forward
        for g, group in sgroup:
            stx_pool[g].extend(group.values)
        
        for b in range(n_blocks):
            for s in range(n_shards):
                tx_pool = stx_pool[s]
                tx_forward = stx_forward[s]
                block_txs = TxBuff(tx_per_block)
                n_forward = min(len(tx_forward), tx_per_block)
                block_txs.extend(tx_forward.popnleft(n_forward))
                n_pool = min(len(tx_pool), tx_per_block-len(block_txs))
                block_txs.extend(tx_pool.popnleft(n_pool))
                self.sblock[s].append(block_txs)
            for s in range(n_shards):
                block = self.sblock[s][-1]
                for tx in block:
                    if abs(tx)!=s:
                        stx_forward[tx].append(-tx)
        observation = [self.stx_pool, self.stx_forward, self.sblock]
        reward = 0
        return observation, reward, 0, None

    def info(self):
        n_block = 0
        n_block_tx = 0
        n_block_out_tx = 0
        n_block_forward_tx = 0
        n_block_inner_tx = 0
        tx_wasted = [0 for i in range(self.n_shards)]
        for s in range(self.n_shards):
            blocks = self.sblock[s]
            n_block += len(blocks)
            for block in blocks:
                n_block_tx += len(block)
                for tx in block:
                    if abs(tx)!=s:
                        n_block_out_tx += 1
                    elif tx>0:
                        n_block_inner_tx += 1
                    else:
                        n_block_forward_tx += 1
                tx_wasted[s] += self.tx_per_block - len(block)
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
            n_tx = self.ptx,
            n_inner_tx = self.n_inner_tx,
            n_cross_tx = self.n_cross_tx,
            prop_cross_tx = float(self.n_cross_tx) / self.ptx,
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            throughput = n_block_tx/self.simulate_time,
            actual_throughput = (n_block_inner_tx+n_block_forward_tx)/self.simulate_time,
            target_throughput = self.tx_per_block*self.n_shards/self.block_interval,
            tx_pool_length = [len(pool) for pool in self.stx_pool],
            tx_forward_length = [len(forward) for forward in self.stx_forward],
            n_wasted = n_wasted,
            tx_wasted = tx_wasted,
            prop_wasted = float(n_wasted) / (n_block * self.tx_per_block)
        )
        return result

class Eth2v5(Eth2v3):
    def step(self, action=None):
        # print('Eth2 v5 step', action)
        # one step contains n_blocks blocks
        n_blocks = self.n_blocks
        allocate = self.allocate
        allocate.apply(action)
        tx_per_block = self.tx_per_block
        block_interval = self.block_interval
        tx_rate = self.tx_rate
        tx_count = block_interval * tx_rate

        # prepare new transactions
        epoch_tx_count = tx_count * n_blocks
        txs = self.txs.iloc[self.ptx:min(self.ptx+epoch_tx_count, len(self.txs))].copy()
        self.ptx += len(txs)
        done = self.ptx >= len(self.txs)

        txs['from_shard'] = txs['from_addr'].map(allocate.allocate) # from shard index
        txs['to_shard'] = txs['to_addr'].map(allocate.allocate) # to shard index

        counts = (txs['from_shard'] == txs['to_shard']).value_counts()
        self.n_inner_tx += counts.get(True, default=0)
        self.n_cross_tx += counts.get(False, default=0)

        # re-allocate tx pool
        stx_pool = [deque() for i in range(self.n_shards)]
        stx_forward = [deque() for i in range(self.n_shards)]
        for shard, (tx_pool, tx_forward) in enumerate(zip(self.stx_pool, self.stx_forward)):
            for from_addr, to_addr, _, _ in tx_pool:
                from_shard = allocate.allocate(from_addr)
                to_shard = allocate.allocate(to_addr)
                stx_pool[from_shard].append((from_addr, to_addr, from_shard, to_shard))
            for from_addr, to_addr, _, _ in tx_forward:
                from_shard = allocate.allocate(from_addr)
                to_shard = allocate.allocate(to_addr)
                stx_forward[to_shard].append((from_addr, to_addr, from_shard, to_shard))
        self.stx_pool = stx_pool
        self.stx_forward = stx_forward

        # start simulation
        for slot_ptx in range(0, len(txs), tx_count):
            # in each time slot, tx_rate new transactions arrived
            slot_txs = txs.iloc[slot_ptx: min(slot_ptx+tx_count, len(txs))]
            for from_shard, tx in slot_txs.groupby('from_shard'):
                #txs store in tuple: (from_addr, to_addr, from_shard, to_shard)
                stx_pool[from_shard].extend(tx.itertuples(index=False, name=None))
            # each shard produce one block
            for shard, (tx_pool, tx_forward) in enumerate(zip(stx_pool, stx_forward)):
                n_forward = min(len(tx_forward), tx_per_block)
                n_pool = min(len(tx_pool), tx_per_block-n_forward)
                block_txs = [tx_forward.popleft() for _ in range(n_forward)]+[tx_pool.popleft() for _ in range(n_pool)]
                self.sblock[shard].append(block_txs)
            # cross shard tx forward to target shard in next timeslot
            for shard, blocks in enumerate(self.sblock):
                for tx in blocks[-1]:
                    to_shard = tx[3]
                    if to_shard!=shard:
                        stx_forward[to_shard].append(tx)
            self.simulate_time += block_interval

        observation = [self.stx_pool, self.stx_forward, self.sblock]
        reward = 0
        return observation, reward, done, None
