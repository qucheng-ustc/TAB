import pandas as pd
import numpy as np
from collections import deque
import multiprocessing as mp

class Protocol:
    # stop simulation
    MSG_TYPE_CTRL_RESET = 0x00
    # control simulation
    MSG_TYPE_CTRL_BLOCK = 0x01
    MSG_TYPE_CTRL_ALLOCATION = 0x02
    MSG_TYPE_CTRL_TRANSITION = 0x03
    MSG_TYPE_CTRL_REPLY = 0x10
    # get info
    MSG_TYPE_CTRL_INFO = 0x04
    MSG_TYPE_CTRL_REPORT = 0x11
    # sync blocks
    MSG_TYPE_CTRL_SYNC_BLOCK = 0x05
    MSG_TYPE_CTRL_SEND_BLOCK = 0x12
    # sync tx pools
    MSG_TYPE_CTRL_SYNC_POOL = 0x06
    MSG_TYPE_CTRL_SEND_POOL = 0x13
    # query latest action
    MSG_TYPE_QUERY_ACTION = 0x07
    MSG_TYPE_QUERY_ACTION_REPLY = 0x14
    # send client txs
    MSG_TYPE_CLIENT_TX = 0x20
    # forward tx between shards
    MSG_TYPE_SHARD_FORWARD_TX = 0x30
    MSG_TYPE_SHARD_TX = 0x31
    # shard local graph
    MSG_TYPE_SHARD_GRAPH = 0x32
    # allocation table
    MSG_TYPE_SHARD_ALLOCATION = 0x33

class LocalGraph:
    def __init__(self, blocks):
        vertex_list = []
        vertex_dict = {}
        edge_table = []
        n_vertex = 0
        n_edge = 0
        for block in blocks:
            for from_addr, to_addr, *_ in block:
                if from_addr not in vertex_dict:
                    vertex_dict[from_addr] = len(vertex_list)
                    vertex_list.append(from_addr)
                    edge_table.append({})
                    n_vertex += 1
                if to_addr not in vertex_dict:
                    vertex_dict[to_addr] = len(vertex_list)
                    vertex_list.append(to_addr)
                    edge_table.append({})
                    n_vertex += 1
                from_idx, to_idx = vertex_dict[from_addr], vertex_dict[to_addr]
                idx_min, idx_max = min(from_idx, to_idx), max(from_idx, to_idx)
                edge_dict = edge_table[idx_min]
                if idx_max not in edge_dict:
                    edge_dict[idx_max] = 1
                    n_edge += 1
                else:
                    edge_dict[idx_max] += 1
        self.vertex_list = vertex_list
        self.edge_weights = np.zeros(shape=n_edge, dtype=np.int)
        self.edge_dests = np.zeros(shape=n_edge, dtype=np.int)
        self.edge_splits = []
        i = 0
        for edge_dict in edge_table:
            for edge_dest, edge_weight in edge_dict.items():
                self.edge_weights[i] = edge_weight
                self.edge_dests[i] = edge_dest
                i += 1
            self.edge_splits.append(i)
    
    def save(self, path='./metis/graphs/local_graph.txt', debug=False):
        self.save_path = path
        n_vertex = len(self.vertex_list)
        n_edge = len(self.edge_weights)
        if debug:
            print(path, n_vertex, n_edge)
        with open(path, 'w') as f:
            f.write(f'{n_vertex} {n_edge}\n')
            i = 0
            for v in range(n_vertex):
                vertex = self.vertex_list[v]
                if isinstance(vertex, str):
                    f.write(f'{vertex}')
                elif isinstance(vertex, tuple):
                    f.write(f'({vertex[0]},{vertex[1]})')
                while i<self.edge_splits[v]:
                    v_next = self.edge_dests[i]
                    weight = self.edge_weights[i]
                    f.write(f' {v_next+1} {weight}')
                    i += 1
                f.write('\n')
        return self
    
    @classmethod
    def load(cls, path, debug=False):
        graph = object.__new__(cls)
        with open(path, 'r') as f:
            header = f.readline().strip()
            header = header.split(' ')
            n_vertex, n_edge = int(header[0]), int(header[1])
            vertex_list = []
            edge_weights = np.zeros(shape=n_edge, dtype=np.int)
            edge_dests = np.zeros(shape=n_edge, dtype=np.int)
            edge_splits = []
            i = 0
            for v, line in enumerate(f):
                data = line.strip().split(' ')
                vertex_addr = data[0].strip('()').split(',')
                vertex_list.append((vertex_addr[0],vertex_addr[1]))
                data = data[1:]
                for k in range(0, len(data), 2):
                    dest, weight = int(data[k])-1, int(data[k+1])
                    edge_weights[i] = weight
                    edge_dests[i] = dest
                    i += 1
                edge_splits.append(i)
        if debug:
            print(path, n_vertex, n_edge)
        graph.vertex_list = vertex_list
        graph.edge_weights = edge_weights
        graph.edge_dests = edge_dests
        graph.edge_splits = edge_splits
        return graph

    def compress(self, weight_limit=1, vweight_limit=1):
        # remove any edge with weight<weight_limit
        # remove any vertex with vweight<vweight_limit
        n_vertex = len(self.vertex_list)
        #n_edge = len(self.edge_weights)
        
        vertex_list = []
        edge_weights = []
        edge_dests = []
        edge_splits = []

        # compute vertex weights
        vweights = np.zeros(shape=n_vertex, dtype=int)
        i = 0
        for v in range(n_vertex):
            while i<self.edge_splits[v]:
                v_next = self.edge_dests[i]
                weight = self.edge_weights[i]
                i += 1
                vweights[v] += weight
                vweights[v_next] += weight
        
        # remove vertexes
        vmap = []
        for v in range(n_vertex):
            vweight = vweights[v]
            if vweight>=vweight_limit:
                vmap.append(len(vertex_list))
                vertex_list.append(self.vertex_list[v])
            else:
                vmap.append(-1)

        # remove edges
        i = 0
        for v in range(n_vertex):
            if vmap[v]<0: # vertex removed, skip
                i = self.edge_splits[v]
                continue
            while i<self.edge_splits[v]:
                v_next = self.edge_dests[i]
                weight = self.edge_weights[i]
                i += 1
                v_next = vmap[v_next]
                if v_next<0: # vertex removed, skip
                    continue
                if weight>=weight_limit:
                    edge_dests.append(v_next)
                    edge_weights.append(weight)
            edge_splits.append(len(edge_weights))
        
        # update graph
        self.vertex_list = vertex_list
        self.vweights = np.zeros(shape=len(vertex_list), dtype=int)
        for v, vn in enumerate(vmap):
            if vn>=0:
                self.vweights[vn] = vweights[v]
        self.edge_dests = edge_dests
        self.edge_weights = edge_weights
        self.edge_splits = edge_splits


class GlobalGraph:
    def __init__(self, local_graphs: list[LocalGraph]):
        vertex_list = []
        vertex_dict = {}
        edge_table = []
        n_vertex = 0
        n_edge = 0
        local_vertex_weights = []
        vertex_weights = []
        for local_graph in local_graphs:
            local_vi_map = []
            local_n_vertex = len(local_graph.vertex_list)
            for vertex in local_graph.vertex_list:
                if vertex not in vertex_dict:
                    vertex_dict[vertex] = len(vertex_list)
                    vertex_list.append(vertex)
                    edge_table.append({})
                    vertex_weights.append(0)
                    local_vertex_weights.append(0)
                    n_vertex += 1
                local_vi_map.append(vertex_dict[vertex])
            i = 0
            if hasattr(local_graph, 'vweights'):
                for local_vi in range(local_n_vertex):
                    vi = local_vi_map[local_vi]
                    local_vertex_weights[vi] = max(local_vertex_weights[vi], local_graph.vweights[local_vi])
            for local_vi in range(local_n_vertex):
                vi = local_vi_map[local_vi]
                edges_vi = edge_table[vi]
                while i<local_graph.edge_splits[local_vi]:
                    edge_weight = local_graph.edge_weights[i]
                    edge_dest = local_graph.edge_dests[i]
                    i += 1
                    vj = local_vi_map[edge_dest]
                    if vi==vj: # connected to self
                        vertex_weights[vi] = max(vertex_weights[vi], 2*edge_weight)
                        continue
                    if vj not in edges_vi:
                        edges_vi[vj] = edge_weight
                        n_edge += 1
                    elif edge_weight>edges_vi[vj]:
                        edges_vi[vj] = edge_weight
                    edges_vj = edge_table[vj]
                    if vi not in edges_vj:
                        edges_vj[vi] = edge_weight
                    elif edge_weight>edges_vj[vi]:
                        edges_vj[vi] = edge_weight
        for v, edge_dict in enumerate(edge_table):
            for weight in edge_dict.values():
                vertex_weights[v] += weight
        for v in range(n_vertex):
            vertex_weights[v] = max(vertex_weights[v], local_vertex_weights[v])
        self.vertex_list = vertex_list
        self.edge_table = edge_table
        self.n_vertex = n_vertex
        self.n_edge = n_edge
        self.vertex_weights = vertex_weights

    def save(self, v_weight=True, path='./metis/graphs/global_graph.txt', debug=False):
        self.save_path = path
        if debug:
            n_edge = 0
            weight_dict = {}
        with open(path, 'w') as f:
            format = '011' if v_weight else '001'
            f.write(f'{self.n_vertex} {self.n_edge} {format}\n')
            for v in range(self.n_vertex):
                space = ''
                if v_weight:
                    f.write(f'{self.vertex_weights[v]}')
                    space = ' '
                for v_next, weight in self.edge_table[v].items():
                    f.write(f'{space}{v_next+1} {weight}')
                    space = ' '
                    if debug:
                        n_edge += 1
                        weight_dict[(min(v,v_next),max(v,v_next))] = weight
                f.write('\n')
        if debug:
            print('Edges:', n_edge, n_edge//2)
            print('Edge weights:', len(weight_dict))
        return self

    def partition(self, n, v_weight=True, save_path='./metis/graphs/global_graph.txt', debug=False):
        if self.n_vertex==0:
            return {}
        self.save(v_weight=v_weight, path=save_path)
        from graph.partition import Partition
        part = Partition(self.save_path)
        parts = part.partition(n, debug=debug)
        partition_table = {a:s for a,s in zip(self.vertex_list,parts)}
        return partition_table

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
    
    def ctrl(self, msg_type, content=None):
        self.broadcast(-1, msg_type=msg_type, content=content)

    def report(self, id, msg_type, content):
        self.report_queue.put((id, msg_type, content))
    
    def query(self, id, msg_type, content=None):
        self.send(-1, id, msg_type, content)

    def get_reply(self):
        return self.report_queue.get()
    
    def join(self, msg_type):
        reply = dict()
        while len(reply)<len(self.queues):
            id, report_type, content = self.get_reply()
            if report_type == msg_type:
                reply[id] = content
        return reply
    
    def close(self):
        for queue in self.queues.values():
            queue.close()
        self.report_queue.close()
        self.closed = True

class ShardSimulator(mp.Process):
    def __init__(self, id: int, net: NetworkSimulator, allocate, tx_per_block, n_blocks, n_shards, shard_allocation, compress, *, daemon=True):
        super().__init__(daemon=daemon)
        self.id = id
        self.net = net
        self.allocate = allocate
        self.tx_per_block = tx_per_block
        self.n_blocks = n_blocks
        self.n_shards = n_shards
        self.shard_allocation = shard_allocation
        self.compress = compress

    def run(self):
        id = self.id
        net = self.net
        allocate = self.allocate
        tx_per_block = self.tx_per_block
        n_blocks = self.n_blocks
        n_shards = self.n_shards
        shard_allocation = self.shard_allocation
        tx_pool = deque()
        tx_forward = deque()
        blocks = []
        if id==0:
            local_graphs = dict()
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
                case Protocol.MSG_TYPE_CTRL_ALLOCATION if shard_allocation:
                    # construct local graph and send to shard 0
                    local_graph = LocalGraph(blocks[max(0,len(blocks)-n_blocks):len(blocks)])
                    if self.compress is not None:
                        local_graph.compress(*self.compress)
                    if id == 0:
                        # collects local graphs and construct global graph, partition it and broadcast account allocation table
                        local_graphs[id] = local_graph
                    else:
                        net.send(id, 0, Protocol.MSG_TYPE_SHARD_GRAPH, local_graph)
                case Protocol.MSG_TYPE_CTRL_ALLOCATION: # use external allocation method
                    allocate.apply(content)
                    net.report(id, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
                case Protocol.MSG_TYPE_SHARD_GRAPH if id == 0:
                    local_graphs[from_id] = content
                    if len(local_graphs)>=n_shards:
                        global_graph = GlobalGraph(local_graphs=local_graphs.values())
                        account_table = global_graph.partition(n_shards)
                        local_graphs = {}
                        net.broadcast(id, Protocol.MSG_TYPE_SHARD_ALLOCATION, account_table)
                case Protocol.MSG_TYPE_QUERY_ACTION:
                    net.report(id, Protocol.MSG_TYPE_QUERY_ACTION_REPLY, account_table) # report latest allocation action to client
                case Protocol.MSG_TYPE_SHARD_ALLOCATION:
                    allocate.apply(content)
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
                case Protocol.MSG_TYPE_CTRL_SYNC_BLOCK:
                    start = content
                    net.report(id, Protocol.MSG_TYPE_CTRL_SEND_BLOCK, blocks[start:])
                case Protocol.MSG_TYPE_CTRL_SYNC_POOL:
                    net.report(id, Protocol.MSG_TYPE_CTRL_SEND_POOL, (tx_pool, tx_forward))

class HarmonySimulator:
    def __init__(self, client, allocate, n_shards, tx_per_block=200, block_interval=15, n_blocks=10, shard_allocation=True, compress=None):
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

        self.shard_allocation = shard_allocation
        self.compress = compress
    
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

        self.shards = [
            ShardSimulator(id, net=self.network, allocate=self.allocate, tx_per_block=self.tx_per_block, n_blocks=self.n_blocks, 
                           n_shards=self.n_shards, shard_allocation=self.shard_allocation, compress=self.compress)
            for id in self.shard_ids]
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
    
    def sync_block(self):
        cur = self.block_height()
        if cur == self.simulate_blocks:
            return
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_SYNC_BLOCK, cur)
        results = self.network.join(Protocol.MSG_TYPE_CTRL_SEND_BLOCK)
        for shard, blocks in results.items():
            self.sblock[shard].extend(blocks)
    
    def sync_pool(self):
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_SYNC_POOL)
        results = self.network.join(Protocol.MSG_TYPE_CTRL_SEND_POOL)
        for shard, (tx_pool, tx_forward) in results.items():
            self.stx_pool[shard] = tx_pool
            self.stx_forward[shard] = tx_forward
    
    def get_block_n_txs(self, start=0, end=None):
        self.sync_block()
        start, end = self._adjust_block_slice(start, end)
        n_txs = [0]*self.n_shards
        for shard,blocks in enumerate(self.sblock):
            for block_id,block in enumerate(blocks[start:end]):
                n_txs[shard] += len(block)
        return n_txs

    def get_block_txs(self, start=0, end=None):
        self.sync_block()
        start, end = self._adjust_block_slice(start, end)
        txs = []
        for shard,blocks in enumerate(self.sblock):
            for block_id,block in enumerate(blocks[start:end]):
                for from_addr, to_addr, _, from_shard, *_ in block:
                    if from_shard == shard: # only return inner txs and out txs
                        txs.append((shard, block_id+start, from_addr, to_addr))
        return pd.DataFrame(txs, columns=['shard', 'block', 'from', 'to'])
    
    def get_pending_txs(self, forward=False):
        self.sync_pool()
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
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_ALLOCATION, action)
        self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
        if self.shard_allocation:
            self.network.query(0, Protocol.MSG_TYPE_QUERY_ACTION)
            id, msg_type, action = self.network.get_reply()
        self.allocate.apply(action)
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_TRANSITION)
        self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
        # self.simulate_time += self.block_interval # add time intervals

        # start simulation
        for _ in range(self.n_blocks):
            slot_txs = self.client.next(time_interval=self.block_interval)
            timestamp = self.simulate_time
            for from_addr, to_addr, *_ in slot_txs.itertuples(index=False, name=None):
                from_shard = self.allocate.allocate(from_addr)
                to_shard = self.allocate.allocate(to_addr)
                #tx store in tuple: (from_addr, to_addr, timestamp, from_shard, to_shard)
                tx = (from_addr, to_addr, timestamp, from_shard, to_shard)
                self.network.send(-1, from_shard, Protocol.MSG_TYPE_CLIENT_TX, tx)
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
