import pandas as pd
import numpy as np
from collections import deque
from collections.abc import Iterable
import multiprocessing as mp
import os
import json
import math
import time
import gc

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
    # sync tx pools
    MSG_TYPE_CTRL_SYNC_POOL = 0x06
    MSG_TYPE_CTRL_SEND_POOL = 0x13
    # query latest action
    MSG_TYPE_QUERY_ACTION = 0x07
    MSG_TYPE_QUERY_ACTION_REPLY = 0x14
    # query cost
    MSG_TYPE_QUERY_COST = 0x08
    MSG_TYPE_QUERY_COST_REPLY = 0x15
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
    def __init__(self, blocks=[]):
        self.vertex_dict = {}
        self.vertex_list = []
        self.edge_table = []
        self.n_edge = 0
        for block in blocks:
            self.append(block)
    
    def append(self, block):
        vertex_dict = self.vertex_dict
        vertex_list = self.vertex_list
        edge_table = self.edge_table
        for from_addr, to_addr, *_ in block:
            if from_addr not in vertex_dict:
                vertex_dict[from_addr] = len(vertex_list)
                vertex_list.append(from_addr)
                edge_table.append({})
            if to_addr not in vertex_dict:
                vertex_dict[to_addr] = len(vertex_list)
                vertex_list.append(to_addr)
                edge_table.append({})
            from_idx, to_idx = vertex_dict[from_addr], vertex_dict[to_addr]
            idx_min, idx_max = min(from_idx, to_idx), max(from_idx, to_idx)
            edge_dict = edge_table[idx_min]
            if idx_max not in edge_dict:
                edge_dict[idx_max] = 1
                self.n_edge += 1
            else:
                edge_dict[idx_max] += 1
    
    def prepare(self): # prepare to send to other shard
        self.edge_weights = np.zeros(shape=self.n_edge, dtype=np.int)
        self.edge_dests = np.zeros(shape=self.n_edge, dtype=np.int)
        self.edge_splits = []
        i = 0
        for edge_dict in self.edge_table:
            for edge_dest, edge_weight in edge_dict.items():
                self.edge_weights[i] = edge_weight
                self.edge_dests[i] = edge_dest
                i += 1
            self.edge_splits.append(i)
        del self.edge_table
        del self.n_edge
        del self.vertex_dict
    
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
        old_partition = {}
        for shard, local_graph in enumerate(local_graphs):
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
                    local_vweight = local_graph.vweights[local_vi]
                    if local_vweight>0:
                        old_partition[vi] = shard
                    local_vertex_weights[vi] = max(local_vertex_weights[vi], local_vweight)
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
        self.old_partition = old_partition

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

    def partition(self, n, v_weight=True, pmatch=False, weight_limit=0, debug=False):
        if self.n_vertex==0:
            return {}
        import mymetis
        xadj = [0]
        adjncy = []
        adjwgt = []
        if v_weight:
            vwgt = []
        else:
            vwgt = None
        i = 0
        for v in range(self.n_vertex):
            if v_weight:
                vwgt.append(self.vertex_weights[v])
            for v_next, weight in self.edge_table[v].items():
                adjncy.append(v_next)
                adjwgt.append(weight)
                i += 1
            xadj.append(i)
        _, parts = mymetis.partition(xadj=xadj, adjncy=adjncy, vwgt=vwgt, adjwgt=adjwgt, nparts=n)
        if pmatch:
            import munkres
            matrix = np.zeros(shape=(n,n), dtype=int)
            orphan = 0
            for v in range(self.n_vertex):
                part = parts[v]
                if v in self.old_partition:
                    old_part = self.old_partition[v]
                    matrix[part,old_part] += 1
                else:
                    # orphan vertex, generated from outbound txs where relay txs are still in tx forward
                    orphan += 1
            # print("Total:", self.n_vertex, "Orphan:", orphan)
            cost_matrix = munkres.make_cost_matrix(matrix)
            m = munkres.Munkres()
            indexes = m.compute(cost_matrix)
            part_match = {}
            for row, column in indexes:
                part_match[row] = column
            # print(indexes)
            partition_table = {a:part_match[s] for a,s in zip(self.vertex_list,parts)}
            # further reduce cost by reserve orphan vertexes and low weight vertexes
            if weight_limit > 0:
                for v, a in enumerate(self.vertex_list):
                    if v not in self.old_partition:
                        # always delete orphan vertex since we can not determine their shard
                        del partition_table[a]
                        orphan -= 1
                    elif self.vertex_weights[v]<weight_limit:
                        partition_table[a] = self.old_partition[v]
                assert(orphan == 0) # check all orphan vertexes are deleted
        else:
            partition_table = {a:s for a,s in zip(self.vertex_list,parts)}
        return partition_table

# helper for calculating overhead
class Overhead:
    def __init__(self, addr_size=40, int_size=32, int32_size=4, int64_size=8, hash_size=32, sign_size=96, tx_size=None, block_header_size=None, account_size=None, record=True):
        self.addr_size = addr_size
        self.int_size = int_size # big int size
        self.int32_size = int32_size
        self.int64_size = int64_size
        self.hash_size = hash_size
        self.sign_size = sign_size
        self.tx_size = tx_size
        self.block_header_size = block_header_size
        self.account_size = account_size
        if self.tx_size is None:
            # default tx:
            #   nonce     int64  8    # account nonce
            #   from      addr   40   # from address
            #   to        addr   40   # to address
            #   value     int    32   # transfer value
            #   timestamp int64  8    # time
            #   price     int    32   # fee price
            #   signature sign   96   # signature
            #   total            256 Bytes
            self.tx_size = 2*self.int64_size + 2*self.int_size + 2*self.addr_size + self.sign_size
        if self.block_header_size is None:
            # default block header:
            #   version          int32  4    # indicates block version & block type
            #   timestamp        int64  8    # time
            #   prev_hash        hash   32   # previous block hash
            #   state_root       hash   32   # root of state tree
            #   tx_root          hash   32   # root of transaction tree
            #   outbound_tx_root hash   32   # root of outbound relay transactions
            #   extend           hash   32   # used when block is prepare block, allocation block, state block
            #   shard_id         int32  4    # shard id
            #   cross_link       hash   32   # cross_link with beacon chain block
            #   extra            hash   32   # additional inner shard consensus informations
            #   signature        sign   96   # accumulated signature
            #   total                   336 Bytes
            self.block_header_size = 2*self.int32_size + self.int64_size + 7*self.hash_size + self.sign_size
            # total size of a shard block is block_header_size + n*tx_size
        if self.account_size is None:
            # default account state:
            #    nonce      int64  8   # account nonce
            #    address    addr   40  # account address
            #    value      int    32  # account value
            #    state_root hash   32  # account state root (optional, because we only consider transfer tx)
            #    total             112 Bytes
            self.account_size = self.int64_size + self.addr_size + self.int_size + self.hash_size
        self.record = record
        self.reset()
    
    def reset(self):
        if self.record:
            self.costs = {
                "state_transition":[],
                "local_graph":[],
                "state_size":[],
                "allocation_table":[],
                "partition_time":[],
            }
    
    def save(self, path, skip=1):
        costs = {k:v[skip:] for k,v in self.costs.items()}
        with open(path, "w") as f:
            json.dump(costs, f)

    def partition_begin(self):
        self.start_time = time.perf_counter()
        return self.start_time
    
    def partition_end(self):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time-self.start_time
        if self.record:
            self.costs['partition_time'].append(elapsed_time)
        return elapsed_time

    def allocation_table_cost(self, allocate):
        account_table = None
        if hasattr(allocate, 'account_table'):
            account_table = allocate.account_table
        elif hasattr(allocate, 'base'):
            if hasattr(allocate.base, 'account_table'):
                account_table = allocate.base.account_table
        if account_table is None:
            return 0
        cost = 0
        for k, v in account_table.items():
            if isinstance(k, tuple):
                cost += 2*self.addr_size+self.int32_size
            else:
                cost += self.addr_size+self.int32_size
        if self.record:
            self.costs['allocation_table'].append(cost)
        return cost

    def state_size_cost(self, n_shards, n_trans):
        proof_size = int(math.ceil(math.log(n_shards)))*self.hash_size
        state_size = n_trans*self.account_size
        cost = proof_size*n_shards*n_shards + state_size
        if self.record:
            self.costs['state_size'].append(cost)
        return cost

    def state_transition_cost(self, allocate, new_table: dict):
        cost = 0
        for a, new_part in new_table.items():
            old_part = allocate.allocate(a)
            if new_part != old_part:
                cost += 1
        if self.record:
            self.costs['state_transition'].append(cost)
        return cost
    
    def local_graph_size(self, local_graph):
        size = 0
        size += len(local_graph.vertex_list)*self.addr_size
        size += len(local_graph.edge_splits)*self.int64_size
        size += len(local_graph.edge_dests)*self.int64_size
        size += len(local_graph.edge_weights)*self.int64_size
        if hasattr(local_graph, "vweights"): # only in compress mode
            size += len(local_graph.vweights)*self.int64_size
        return size
    
    def local_graph_cost(self, local_graphs: list[LocalGraph]):
        cost = 0
        for g in local_graphs:
            cost += self.local_graph_size(g)
        if self.record:
            self.costs['local_graph'].append(cost)
        return cost

class NetworkSimulator:
    # simple full connected network without bandwith limit
    def __init__(self, nodes:list, delay:float=0.):
        # network delay is not implemented, we assume all messages can arrive in one block interval
        self.n_nodes = len(nodes)
        self.nodes = nodes
        self.queues = {idx:mp.Queue() for idx in nodes} # for msg exchange between nodes
        self.report_queue = mp.Queue() # for reporting shard info to main simulator
        self.closed = False

    def send(self, idx, to_idx, msg_type, content=None):
        self.queues[to_idx].put((idx, msg_type, content))
    
    def broadcast(self, idx, msg_type, content=None):
        for queue in self.queues.values():
            queue.put((idx, msg_type, content))

    def recv(self, idx):
        return self.queues[idx].get()
    
    def ctrl(self, msg_type, content=None):
        self.broadcast(-1, msg_type=msg_type, content=content)

    def report(self, idx, msg_type, content=None):
        self.report_queue.put((idx, msg_type, content))
    
    def query(self, idx, msg_type, content=None):
        self.send(-1, idx, msg_type, content)

    def get_reply(self):
        return self.report_queue.get()
    
    def join(self, msg_type=None):
        reply = dict()
        while len(reply)<len(self.queues):
            idx, report_type, content = self.get_reply()
            if msg_type is None or report_type == msg_type:
                reply[idx] = content
        return reply
    
    def close(self):
        for queue in self.queues.values():
            queue.close()
        self.report_queue.close()
        self.closed = True

class ShardSimulator(mp.Process):
    def __init__(self, idx: int, net: NetworkSimulator, allocate, tx_per_block, n_blocks, n_shards, shard_allocation, compress=None, pmatch=False, overhead:Overhead|None=None, save_path=None, *, daemon=True):
        super().__init__(daemon=daemon)
        self.idx = idx
        self.net = net
        self.allocate = allocate
        self.tx_per_block = tx_per_block
        self.n_blocks = n_blocks
        self.n_shards = n_shards
        self.shard_allocation = shard_allocation
        self.compress = compress
        self.pmatch = pmatch
        self.overhead = overhead
        self.save_path = save_path
        self.weight_limit = 0
        if self.compress is not None:
            if len(self.compress)>2:
                self.weight_limit = self.compress[2]
        
        self.blocks_path = os.path.join(self.save_path,f'{self.idx}_blocks.txt')
        with open(self.blocks_path, 'w') as file:
            file.truncate(0)
        self.block_height = 0
    
    def save_block(self, block):
        with open(self.blocks_path, 'a') as f:
            for tx_id, tx in enumerate(block):
                # tx : from_addr, to_addr, timestamp, from_shard, to_shard, block_height
                f.write(f'{self.block_height}|{tx_id}|{tx[0]}|{tx[1]}|{tx[2]}|{tx[3]}|{tx[4]}|{tx[5]}\n')

    def save_info(self, tx_pool, tx_forward):
        tx_pool_path = os.path.join(self.save_path,f'{self.idx}_tx_pool.txt')
        with open(tx_pool_path, 'w') as f:
            for tx_id, tx in enumerate(tx_pool):
                # tx = (from_addr, to_addr, timestamp, from_shard, to_shard)
                f.write(f'{tx_id}|{tx[0]}|{tx[1]}|{tx[2]}|{tx[3]}|{tx[4]}\n')
        tx_forward_path = os.path.join(self.save_path,f'{self.idx}_tx_forward.txt')
        with open(tx_forward_path, 'w') as f:
            for tx_id, tx in enumerate(tx_forward):
                # tx : from_addr, to_addr, timestamp, from_shard, to_shard, block_height
                f.write(f'{tx_id}|{tx[0]}|{tx[1]}|{tx[2]}|{tx[3]}|{tx[4]}|{tx[5]}\n')
        if self.overhead is not None:
            overhead_path = os.path.join(self.save_path,f'{self.idx}_overhead.txt')
            self.overhead.save(overhead_path)

    def run(self):
        np.random.seed(0)
        idx = self.idx
        net = self.net
        allocate = self.allocate
        tx_per_block = self.tx_per_block
        n_shards = self.n_shards
        shard_allocation = self.shard_allocation
        tx_pool = deque()
        tx_forward = deque()
        # blocks = []
        local_graph = LocalGraph()
        if idx==0:
            local_graphs = dict()
        while True:
            from_idx, msg_type, content = net.recv(idx)
            match msg_type:
                case Protocol.MSG_TYPE_CTRL_RESET:
                    self.save_info(tx_pool, tx_forward)
                    net.report(idx, Protocol.MSG_TYPE_CTRL_REPORT)
                    return
                case Protocol.MSG_TYPE_CTRL_INFO:
                    self.save_info(tx_pool, tx_forward)
                    net.report(idx, Protocol.MSG_TYPE_CTRL_REPORT)
                case Protocol.MSG_TYPE_CTRL_BLOCK:
                    # each shard produce one block
                    block_height = self.block_height
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
                        if to_shard!=idx:
                            net.send(idx, to_shard, Protocol.MSG_TYPE_SHARD_FORWARD_TX, tx)
                    local_graph.append(block_txs)
                    self.save_block(block_txs)
                    # delete block_txs after save it to file
                    del block_txs
                    self.block_height += 1
                    net.report(idx, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
                case Protocol.MSG_TYPE_SHARD_FORWARD_TX:
                    tx = content
                    tx_forward.append(tx)
                case Protocol.MSG_TYPE_CTRL_ALLOCATION if shard_allocation:
                    # construct local graph and send to shard 0
                    local_graph.prepare()
                    if self.compress is not None:
                        local_graph.compress(self.compress[0], self.compress[1])
                        # set vertex weight that are not in this shard to 0
                        for vertex_idx, vertex in enumerate(local_graph.vertex_list):
                            vertex_shard = allocate.allocate(vertex)
                            if vertex_shard != idx:
                                local_graph.vweights[vertex_idx] = 0
                    if idx == 0:
                        # collects local graphs and construct global graph, partition it and broadcast account allocation table
                        local_graphs[idx] = local_graph
                    else:
                        net.send(idx, 0, Protocol.MSG_TYPE_SHARD_GRAPH, local_graph)
                case Protocol.MSG_TYPE_CTRL_ALLOCATION: # use external allocation method
                    allocate.apply(content)
                    if idx == 0 and self.overhead is not None:
                        self.overhead.allocation_table_cost(allocate)
                    net.report(idx, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
                case Protocol.MSG_TYPE_SHARD_GRAPH if idx == 0:
                    local_graphs[from_idx] = content
                    if len(local_graphs)>=n_shards:
                        if self.overhead is not None:
                            self.overhead.local_graph_cost(local_graphs.values())
                            self.overhead.partition_begin()
                        global_graph = GlobalGraph(local_graphs=local_graphs.values())
                        account_table = global_graph.partition(n_shards, pmatch=self.pmatch, weight_limit=self.weight_limit)
                        if self.overhead is not None:
                            self.overhead.partition_end()
                            n_trans = self.overhead.state_transition_cost(allocate=allocate, new_table=account_table)
                            self.overhead.state_size_cost(n_shards=n_shards, n_trans=n_trans)
                        local_graphs = {}
                        net.broadcast(idx, Protocol.MSG_TYPE_SHARD_ALLOCATION, account_table)
                case Protocol.MSG_TYPE_QUERY_ACTION:
                    net.report(idx, Protocol.MSG_TYPE_QUERY_ACTION_REPLY, account_table) # report latest allocation action to client
                case Protocol.MSG_TYPE_QUERY_COST:
                    net.report(idx, Protocol.MSG_TYPE_QUERY_COST_REPLY, self.overhead)
                case Protocol.MSG_TYPE_SHARD_ALLOCATION:
                    local_graph = LocalGraph() # reset local graph
                    allocate.apply(content)
                    if idx == 0 and self.overhead is not None:
                        self.overhead.allocation_table_cost(allocate)
                    net.report(idx, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
                case Protocol.MSG_TYPE_CTRL_TRANSITION:
                    # gc before transition
                    gc.collect()
                    # re-allocate tx pool
                    new_tx_pool = deque()
                    new_tx_forward = deque()
                    for from_addr, to_addr, timestamp, *_ in tx_pool:
                        from_shard = allocate.allocate(from_addr)
                        to_shard = allocate.allocate(to_addr)
                        tx = (from_addr, to_addr, timestamp, from_shard, to_shard)
                        if from_shard != idx:
                            net.send(idx, from_shard, Protocol.MSG_TYPE_SHARD_TX, tx)
                        else:
                            new_tx_pool.append(tx)
                    tx_pool = new_tx_pool
                    for from_addr, to_addr, timestamp, _, _, block_height in tx_forward:
                        from_shard = allocate.allocate(from_addr)
                        to_shard = allocate.allocate(to_addr)
                        tx = (from_addr, to_addr, timestamp, from_shard, to_shard, block_height)
                        if to_shard != idx:
                            net.send(idx, to_shard, Protocol.MSG_TYPE_SHARD_FORWARD_TX, tx)
                        else:
                            new_tx_forward.append(tx)
                    tx_forward = new_tx_forward
                    net.report(idx, Protocol.MSG_TYPE_CTRL_REPLY, msg_type)
                case Protocol.MSG_TYPE_SHARD_TX | Protocol.MSG_TYPE_CLIENT_TX:
                    tx = content
                    tx_pool.append(tx)
                case Protocol.MSG_TYPE_CTRL_SYNC_POOL:
                    net.report(idx, Protocol.MSG_TYPE_CTRL_SEND_POOL, (tx_pool, tx_forward))

class HarmonySimulator:
    def __init__(self, client, allocate, n_shards, tx_per_block=200, block_interval=15, n_blocks=10, shard_allocation=True, compress=None, pmatch=False, overhead=None, save_path=None):
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
        self.pmatch = pmatch
        self.overhead = overhead
        self.save_path = save_path

        self.is_closed = True
    
    def close(self):
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_RESET)
        self.network.join(Protocol.MSG_TYPE_CTRL_REPORT)
        for s in self.shards:
            s.terminate()
            s.join()
            s.close()
        self.network.close()
        self.is_closed = True

    def reset(self, ptx=0):
        self.allocate.reset()
        self.client.reset(ptx=ptx)
        self.simulate_time = 0
        self.block_height = 0
        self.n_epochs = 0

        self.stx_pool = [deque() for _ in range(self.n_shards)]
        self.stx_forward = [deque() for _ in range(self.n_shards)]

        if self.network is not None and not self.network.closed:
            self.close()
        self.shard_ids = list(range(self.n_shards))
        self.network = NetworkSimulator(self.shard_ids)
        
        shard_simulator = ShardSimulator(0, net=self.network, allocate=self.allocate, tx_per_block=self.tx_per_block, n_blocks=self.n_blocks, 
                           n_shards=self.n_shards, shard_allocation=self.shard_allocation, compress=self.compress, pmatch=self.pmatch, overhead=self.overhead, save_path=self.save_path)
        self.shards = [shard_simulator]
        for idx in self.shard_ids[1:]:
            shard_simulator = ShardSimulator(idx, net=self.network, allocate=self.allocate, tx_per_block=self.tx_per_block, n_blocks=self.n_blocks, 
                           n_shards=self.n_shards, shard_allocation=self.shard_allocation, compress=self.compress, pmatch=self.pmatch, overhead=None, save_path=self.save_path)
            self.shards.append(shard_simulator)
        for s in self.shards: s.start()
        self.is_closed = False
        return self.client.done(time_interval=self.epoch_time)
    
    def _adjust_block_slice(self, start, end):
        block_height = self.block_height
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
    
    def sync_pool(self):
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_SYNC_POOL)
        results = self.network.join(Protocol.MSG_TYPE_CTRL_SEND_POOL)
        for shard, (tx_pool, tx_forward) in results.items():
            self.stx_pool[shard] = tx_pool
            self.stx_forward[shard] = tx_forward
    
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

        # Note: all shards perform allocation algorithm in the paper so allocation table do not need to be broadcast. We only execute allocation algorithm in one shard here for fast simulation on single machine thus we broadcast the result table.

        # Allocation & Transition, client is shutdown during this period
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_ALLOCATION, action)
        self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
        if self.shard_allocation:
            self.network.query(0, Protocol.MSG_TYPE_QUERY_ACTION)
            idx, msg_type, action = self.network.get_reply()
        self.allocate.apply(action)
        self.network.ctrl(Protocol.MSG_TYPE_CTRL_TRANSITION)
        self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
        # self.simulate_time += self.block_interval # add idle time, do not need. Because in the paper, transaction processing are not shutdown during allocation phase which introduce no idle time.

        # start simulation
        for _ in range(self.n_blocks):
            slot_txs = self.client.next(time_interval=self.block_interval)
            timestamp = self.simulate_time
            for from_addr, to_addr, *_ in slot_txs:
                from_shard = self.allocate.allocate(from_addr)
                to_shard = self.allocate.allocate(to_addr)
                #tx store in tuple: (from_addr, to_addr, timestamp, from_shard, to_shard)
                tx = (from_addr, to_addr, timestamp, from_shard, to_shard)
                self.network.send(-1, from_shard, Protocol.MSG_TYPE_CLIENT_TX, tx)
                timestamp += 1./self.tx_rate
            self.network.ctrl(Protocol.MSG_TYPE_CTRL_BLOCK)
            self.network.join(Protocol.MSG_TYPE_CTRL_REPLY)
            self.simulate_time += self.block_interval
            self.block_height += 1
        self.n_epochs += 1
        return self.client.done(time_interval=self.epoch_time)

    def info(self, start=0, end=None):
        if not self.is_closed:
            # force shards save info
            self.network.ctrl(Protocol.MSG_TYPE_CTRL_INFO)
            self.network.join(Protocol.MSG_TYPE_CTRL_REPORT)
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
        tx_pool_length = []
        tx_forward_length = []
        for shard in range(self.n_shards):
            blocks_path = os.path.join(self.save_path,f'{shard}_blocks.txt')
            blocks = []
            with open(blocks_path, 'r') as f:
                for line in f:
                    items = line.split('|')
                    # block_id, tx_id, from_addr, to_addr, timestamp, from_shard, to_shard, block_height
                    block_id = int(items[0])
                    tx_id = int(items[1])
                    from_addr = items[2]
                    to_addr = items[3]
                    timestamp = float(items[4])
                    from_shard = int(items[5])
                    to_shard = int(items[6])
                    block_height = int(items[7])
                    while block_id>=len(blocks):
                        blocks.append([])
                    blocks[block_id].append((from_addr, to_addr, timestamp, from_shard, to_shard, block_height))
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
            tx_pool_path = os.path.join(self.save_path,f'{shard}_tx_pool.txt')
            with open(tx_pool_path, 'r') as f:
                tx_pool_length.append(sum(1 for line in f))
            tx_forward_path = os.path.join(self.save_path,f'{shard}_tx_forward.txt')
            with open(tx_forward_path, 'r') as f:
                tx_forward_length.append(sum(1 for line in f))
        n_wasted = sum(tx_wasted)
        
        result = dict(
            n_shards = self.n_shards,
            blocks_per_epoch = self.n_blocks,
            tx_rate = self.tx_rate,
            tx_per_block = self.tx_per_block,
            block_interval = self.block_interval,
            max_epochs = self.max_epochs,
            simulate_time = self.simulate_time,
            block_height = self.block_height,
            n_epochs = self.n_epochs,
            n_tx = n_tx,
            start_block = start,
            end_block = end,
            start_time = start_time,
            end_time = end_time,
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
            tx_pool_length = tx_pool_length,
            tx_forward_length = tx_forward_length,
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
        if self.overhead is not None:
            overhead_path = os.path.join(self.save_path,f'0_overhead.txt')
            with open(overhead_path, 'r') as f:
                self.overhead.costs = json.load(f)
            for cost_name, cost_value in self.overhead.costs.items():
                result[f'{cost_name}_cost'] = cost_value
                if isinstance(cost_value, Iterable):
                    result[f'{cost_name}_total_cost'] = sum(cost_value)
        return result
