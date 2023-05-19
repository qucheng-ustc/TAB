import pandas as pd
import numpy as np
from collections import deque
import multiprocessing as mp

class Protocol:
    MSG_TYPE_CTRL_RESET = 0x00

    MSG_TYPE_CTRL_BLOCK = 0x01
    MSG_TYPE_CTRL_ALLOCATION = 0x02
    MSG_TYPE_CTRL_TRANSITION = 0x03
    MSG_TYPE_CTRL_REPLY = 0x10

    MSG_TYPE_CTRL_INFO = 0x04
    MSG_TYPE_CTRL_REPORT = 0x11

    MSG_TYPE_CTRL_SYNC = 0x05
    MSG_TYPE_CTRL_ACK = 0x12

    MSG_TYPE_CLIENT_TX = 0x20

    MSG_TYPE_SHARD_FORWARD_TX = 0x30
    MSG_TYPE_SHARD_TX = 0x31

class NetworkSimulator:
    # simple full connected network
    def __init__(self, nodes:list, delay:float=0.):
        self.n_nodes = len(nodes)
        self.nodes = nodes
        self.queues = {id:mp.Queue() for id in nodes} # for msg exchange between nodes
        self.report_queue = mp.Queue() # for reporting shard info to main simulator
        self.closed = False

    def send(self, id, to_id, msg_type, content):
        self.queues[to_id].put((id, msg_type, content))
    
    def broadcast(self, id, msg_type, content):
        for queue in self.queues.values():
            queue.put((id, msg_type, content))

    def recv(self, id):
        return self.queues[id].get()
    
    def ctrl(self, msg_type):
        self.broadcast(-1, msg_type=msg_type, content=None)

    def report(self, id, msg_type, content):
        self.report_queue.put((id, msg_type, content))
    
    def join(self, msg_type):
        reply = dict()
        while len(reply)<len(self.queues):
            id, report_type, content = self.report_queue.get()
            if report_type == msg_type:
                reply[id] = content
        return reply
    
    def close(self):
        for queue in self.queues.values():
            queue.close()
        self.report_queue.close()
        self.closed = True

def shard_worker(id: int, net: NetworkSimulator, allocate, tx_per_block):
    tx_pool = deque()
    tx_forward = deque()
    blocks = []
    while True:
        from_id, msg_type, content = net.recv(id)
        match msg_type:
            case Protocol.MSG_TYPE_CTRL_RESET:
                return
            case Protocol.MSG_TYPE_CTRL_INFO:
                net.report(id, Protocol.MSG_TYPE_CTRL_REPORT, (blocks, tx_pool, tx_forward))
            case Protocol.MSG_TYPE_CTRL_BLOCK:
                # each shard produce one block
                block_height = len(blocks)
                block_txs = []
                while len(block_txs)<tx_per_block and len(tx_forward)>0:
                    if tx_forward[0][5]>=block_height:
                        break
                    # only append committed forward tx
                    block_txs.append(tx_forward.popleft())
                while len(block_txs)<tx_per_block and len(tx_pool)>0:
                    from_addr, to_addr, timestamp, from_shard, to_shard = tx_pool.popleft()
                    tx = (from_addr, to_addr, timestamp, from_shard, to_shard, block_height)
                    block_txs.append(tx)
                    # cross shard tx forward to target shard
                    if to_shard!=id:
                        net.send(id, to_shard, Protocol.MSG_TYPE_SHARD_FORWARD_TX, tx)
                blocks.append(block_txs)
                net.report(id, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
            case Protocol.MSG_TYPE_SHARD_FORWARD_TX:
                tx = content
                tx_forward.append(tx)
            case Protocol.MSG_TYPE_CTRL_ALLOCATION:
                net.report(id, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
            case Protocol.MSG_TYPE_CTRL_TRANSITION:
                # re-allocate tx pool
                new_tx_pool = deque()
                new_tx_forward = deque()
                for from_addr, to_addr, timestamp, *_ in tx_pool:
                    from_shard = allocate.allocate(from_addr)
                    to_shard = allocate.allocate(to_addr)
                    tx = (from_addr, to_addr, timestamp, from_shard, to_shard)
                    if from_shard != id:
                        net.send(id, from_shard, Protocol.MSG_TYPE_SHARD_TX, tx)
                    else:
                        new_tx_pool.append(tx)
                tx_pool = new_tx_pool
                for from_addr, to_addr, timestamp, _, _, block_height in tx_forward:
                    from_shard = allocate.allocate(from_addr)
                    to_shard = allocate.allocate(to_addr)
                    tx = (from_addr, to_addr, timestamp, from_shard, to_shard, block_height)
                    if to_shard != id:
                        net.send(id, to_shard, Protocol.MSG_TYPE_SHARD_FORWARD_TX, tx)
                    else:
                        new_tx_forward.append(tx)
                tx_forward = new_tx_forward
                net.report(id, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
            case Protocol.MSG_TYPE_SHARD_TX | Protocol.MSG_TYPE_CLIENT_TX:
                tx = content
                tx_pool.append(tx)

class ShardSimulator:
    def __init__(self, id: int, net: NetworkSimulator, allocate, tx_per_block):
        self.id = id
        self.net = net
        self.proc = mp.Process(target=shard_worker, kwargs=dict(id=id, net=net, allocate=allocate, tx_per_block=tx_per_block))

    def start(self):
        self.proc.start()
    
    def join(self):
        self.proc.join()

    def close(self):
        self.proc.close()

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

        self.network = None
        self.shards = []

    def close(self):
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_RESET)
        for s in self.shards: s.join()
        for s in self.shards: s.close()
        self.network.close()

    def reset(self, ptx=0):
        self.allocate.reset()
        self.client.reset(ptx=ptx)
        self.simulate_time = 0
        self.simulate_blocks = 0

        self.stx_pool = [deque() for _ in range(self.n_shards)]
        self.stx_forward = [deque() for _ in range(self.n_shards)]
        self.sblock = [[] for _ in range(self.n_shards)]

        if self.network is not None and not self.network.closed:
            self.close()
        self.shard_ids = list(range(self.n_shards))
        self.network = NetworkSimulator(self.shard_ids)

        self.shards = [ShardSimulator(id, net=self.network, allocate=self.allocate, tx_per_block=self.tx_per_block) for id in self.shard_ids]
        for s in self.shards: s.start()
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
                for from_addr, to_addr, _, from_shard, *_ in block:
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
        # 1. Allocation: Shard #0 collects account graph from other shard, construct whole graph and partition it, create an allocation block and broadcasts allocation block to all shards
        # 2. Transition: All shards apply the account allocation action and start state transition and forward pending txs in tx pool to destination shards, all shards produce state block after receive complete account states
        # 3. Block: All shards process transactions and produce blocks

        # Allocation & Transition, client is shutdown during this period
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_ALLOCATION)
        self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
        self.simulate_time += self.block_interval # simply add intervals
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_TRANSITION)
        self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
        self.simulate_time += self.block_interval # simply add intervals

        # start simulation
        for _ in range(self.n_blocks):
            slot_txs = self.client.next(time_interval=self.block_interval)
            timestamp = self.simulate_time
            for from_addr, to_addr, *_ in slot_txs.itertuples(index=False, name=None):
                from_shard = self.allocate.allocate(from_addr)
                to_shard = self.allocate.allocate(to_addr)
                #tx store in tuple: (from_addr, to_addr, timestamp, from_shard, to_shard)
                tx = (from_addr, to_addr, timestamp, from_shard, to_shard)
                self.network.send(-1, to_shard, Protocol.MSG_TYPE_CLIENT_TX, tx)
                timestamp += 1./self.tx_rate
            self.network.ctrl(Protocol.MSG_TYPE_CTRL_BLOCK)
            self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
            self.simulate_time += self.block_interval
            self.simulate_blocks += 1
        return self.client.done(time_interval=self.epoch_time)

    def info(self, start=0, end=None):
        # re-sync shard blocks & tx pools
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_INFO)
        results = self.network.join(Protocol.MSG_TYPE_CTRL_REPORT)
        for shard, result in results.items():
            blocks, tx_pool, tx_forward = result
            self.sblock[shard] = blocks
            self.stx_pool[shard] = tx_pool
            self.stx_forward[shard] = tx_forward
        start, end = self._adjust_block_slice(start, end)
        n_tx = self.client.n_tx()
        n_block = 0
        n_block_tx = 0
        n_block_out_tx = 0
        n_block_forward_tx = 0
        n_block_inner_tx = 0
        tx_wasted = [0 for _ in range(self.n_shards)]
        tx_delay = []
        start_time = start*self.block_interval
        end_time = end*self.block_interval
        total_time = end_time - start_time
        for shard, blocks in enumerate(self.sblock):
            shard_blocks = blocks[start:end]
            n_block += len(shard_blocks)
            block_time = start_time
            for block in shard_blocks:
                block_time += self.block_interval
                n_block_tx += len(block)
                for _, _, timestamp, from_shard, to_shard, *_ in block:
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
            n_tx = n_tx,
            total_time = total_time,
            n_block = n_block,
            target_n_block = self.n_shards*total_time/self.block_interval,
            n_block_tx = n_block_tx,
            n_block_out_tx = n_block_out_tx,
            n_block_forward_tx = n_block_forward_tx,
            n_block_inner_tx = n_block_inner_tx,
            prop_cross_tx = n_block_out_tx / (n_block_out_tx+n_block_inner_tx) if n_block_tx>0 else 0,
            throughput = n_block_tx/total_time if total_time>0 else 0,
            actual_throughput = (n_block_inner_tx+n_block_forward_tx)/total_time if total_time>0 else 0,
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
