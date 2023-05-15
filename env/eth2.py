import pandas as pd
import numpy as np
import itertools
from collections import deque

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

    def block_height(self):
        return max([len(blocks) for blocks in self.sblock])
    
    def _adjust_block_slice(self, start, end):
        block_height = self.block_height()
        if start<0:
            start = block_height + start
        if end is not None:
            if end<0:
                end = block_height + end
        else:
            end = block_height
        if start<0: start = 0
        if end<0: end = 0
        assert(start<block_height)
        assert(end<=block_height)
        return start, end
    
    def get_block_n_txs(self, start=0, end=None):
        start, end = self._adjust_block_slice(start, end)
        n_txs = [0]*self.n_shards
        for shard,blocks in enumerate(self.sblock):
            for block_id,block in enumerate(blocks[start:end]):
                n_txs[shard] += len(block)
        return n_txs

    def get_block_txs(self, start=0, end=None):
        start, end = self._adjust_block_slice(start, end)
        txs = []
        for shard,blocks in enumerate(self.sblock):
            for block_id,block in enumerate(blocks[start:end]):
                for from_addr, to_addr, _, from_shard, _ in block:
                    if from_shard == shard: # only return inner txs and out txs
                        txs.append((shard, block_id+start, from_addr, to_addr))
        return pd.DataFrame(txs, columns=['shard', 'block', 'from', 'to'])
    
    def get_pending_txs(self, forward=False):
        txs = []
        for shard, shard_tx_pool in enumerate(self.stx_pool):
            for from_addr, to_addr, *_ in shard_tx_pool:
                txs.append((shard, from_addr, to_addr))
        if forward:
            for shard, shard_tx_forward in enumerate(self.stx_forward):
                for from_addr, to_addr, *_ in shard_tx_forward:
                    txs.append((shard, from_addr, to_addr))
        return pd.DataFrame(txs, columns=['shard', 'from', 'to'])

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
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            prop_cross_tx = n_block_out_tx / (n_block_out_tx+n_block_inner_tx) if n_block_tx>0 else 0,
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
        self.epoch_txs = txs
        
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
            for from_addr, to_addr, gas in slot_txs.itertuples(index=False, name=None):
                from_shard = self.allocate.allocate(from_addr)
                to_shard = self.allocate.allocate(to_addr)
                #tx store in tuple: (from_addr, to_addr, gas, from_shard, to_shard)
                stx_pool[from_shard].append((from_addr, to_addr, gas, from_shard, to_shard))
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
        return self.client.done(time_interval=self.epoch_time)

class Eth2v3Simulator(Eth2v1Simulator):
    def step(self, action):
        self.allocate.apply(action) # apply allocate action before txs arrives

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
        for _ in range(self.n_blocks):
            slot_txs = self.client.next(time_interval=self.block_interval)
            timestamp = self.simulate_time
            for from_addr, to_addr, *_ in slot_txs.itertuples(index=False, name=None):
                from_shard = self.allocate.allocate(from_addr)
                to_shard = self.allocate.allocate(to_addr)
                #tx store in tuple: (from_addr, to_addr, timestamp, from_shard, to_shard)
                stx_pool[from_shard].append((from_addr, to_addr, timestamp, from_shard, to_shard))
                timestamp += 1./self.tx_rate
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
        return self.client.done(time_interval=self.epoch_time)

    def info(self):
        n_tx = self.client.n_tx()
        n_block = 0
        n_block_tx = 0
        n_block_out_tx = 0
        n_block_forward_tx = 0
        n_block_inner_tx = 0
        tx_wasted = [0 for _ in range(self.n_shards)]
        tx_delay = []
        for shard, blocks in enumerate(self.sblock):
            n_block += len(blocks)
            block_time = 0.
            for block in blocks:
                block_time += self.block_interval
                n_block_tx += len(block)
                for _, _, timestamp, from_shard, to_shard in block:
                    if to_shard!=shard:
                        n_block_out_tx += 1
                    elif from_shard==shard:
                        n_block_inner_tx += 1
                        tx_delay.append(block_time-timestamp)
                    else:
                        n_block_forward_tx += 1
                        tx_delay.append(block_time-timestamp)
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
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            prop_cross_tx = n_block_out_tx / (n_block_out_tx+n_block_inner_tx) if n_block_tx>0 else 0,
            throughput = n_block_tx/self.simulate_time if self.simulate_time>0 else 0,
            actual_throughput = (n_block_inner_tx+n_block_forward_tx)/self.simulate_time if self.simulate_time>0 else 0,
            target_throughput = self.tx_per_block*self.n_shards/self.block_interval,
            tx_pool_length = [len(pool) for pool in self.stx_pool],
            tx_forward_length = [len(forward) for forward in self.stx_forward],
            n_wasted = n_wasted,
            tx_wasted = tx_wasted,
            prop_wasted = float(n_wasted) / (n_block * self.tx_per_block),
            complete_txs = len(tx_delay),
            tx_delay = sum(tx_delay)/len(tx_delay)
        )
        result['prop_throughput'] = float(result['actual_throughput'])/result['target_throughput']
        result['pool_length_mean'] = np.average(result['tx_pool_length'])
        result['pool_length_std'] = np.std(result['tx_pool_length'])
        result['forward_length_mean'] = np.average(result['tx_forward_length'])
        result['forward_length_std'] = np.std(result['tx_forward_length'])
        result['tx_pending_length'] = [p+f for p,f in zip(result['tx_pool_length'],result['tx_forward_length'])]
        result['pending_length_mean'] = np.average(result['tx_pending_length'])
        result['pending_length_std'] = np.std(result['tx_pending_length'])
        result['wasted_mean'] = np.average(result['tx_wasted'])
        result['wasted_std'] = np.std(result['tx_wasted'])
        return result

class HarmonySimulator:
    def __init__(self):
        pass
