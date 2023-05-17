import pandas as pd
import numpy as np
from collections import deque
import multiprocessing as mp

class Protocol:
    MSG_TYPE_CTRL_RESET = 0x00
    MSG_TYPE_SIM_BLOCK = 0x01
    MSG_TYPE_SIM_REALLOCATE = 0x02


class NetworkSimulator:
    # simple full connected network
    def __init__(self, nodes:list, delay:float=0):
        self.n_nodes = len(nodes)
        self.nodes = nodes
        self.queues = {id:mp.Queue() for id in nodes}
        self.info_queue = mp.Queue() # for reporting information to main simulator

    def send(self, id, to_id, msg_type, content):
        self.queues[to_id].put((id, msg_type, content))

    def recv(self, id):
        return self.queues[id].get()
    
    def report(self, id, msg_type, content):
        self.info_queue.put((id, msg_type, content))
    
    def close(self):
        for queue in self.queues:
            queue.close()
        self.info_queue.close()

def shard_worker(id: int, net: NetworkSimulator, allocate):
    tx_pool = deque()
    tx_forward = deque()
    blocks = []
    done = False
    while not done:
        from_id, msg_type, msg = net.recv(id)
        match msg_type:
            case Protocol.MSG_TYPE_SIM_BLOCK:
                pass

class ShardSimulator:
    def __init__(self, id: int, net: NetworkSimulator, allocate):
        self.id = id
        self.net = net
        self.proc = mp.Process(target=shard_worker, kwargs=dict(id=id, net=net, allocate=allocate))

    def start(self):
        self.proc.start()
    
    def join(self):
        self.proc.join()

    def close(self):
        self.proc.close()
    
    def reset(self):
        self.net.send(id=self.id, to_id=self.id, msg_type=Protocol.MSG_TYPE_CTRL_RESET)

class HarmonySimulator:
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

        self.network_simulator = None
        self.shard_simulators = []

    def reset(self, ptx=0):
        self.allocate.reset()
        self.client.reset(ptx=ptx)
        self.simulate_time = 0

        self.stx_pool = [deque() for _ in range(self.n_shards)]
        self.stx_forward = [deque() for _ in range(self.n_shards)]
        self.sblock = [[] for _ in range(self.n_shards)]

        if self.network_simulator is not None:
            self.network_simulator.close()
        self.shard_ids = list(range(self.n_shards))
        self.network_simulator = NetworkSimulator(self.shard_ids)
        for s in self.shard_simulators: s.reset()
        for s in self.shard_simulators: s.join()
        for s in self.shard_simulators: s.close()
        self.shard_simulators = [ShardSimulator(id, self.network_simulator) for id in self.shard_ids]
        for s in self.shard_simulators: s.start()
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

    def step(self, action):
        # shard #0 collects block txs from other shard
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