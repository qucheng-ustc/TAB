import numpy as np
import pandas as pd

class Graph:
    def __init__(self, txs):
        self.txs = txs
        vertex_idx_from = pd.Index(txs['from'].unique())
        vertex_idx_to = pd.Index(txs['to'].unique())
        print('Vertex from:', len(vertex_idx_from), 'Vertex to:', len(vertex_idx_to))
        self.vertex_idx = vertex_idx_from.union(vertex_idx_to)
        self.n_vertex = len(self.vertex_idx)
        print('Vertex:', self.n_vertex)
        self.nexts = dict()
        self.weights = dict()
        self.v_weights = np.zeros(shape=self.n_vertex, dtype=int)
        txs = txs[['from','to']]
        for index, addr_from, addr_to in txs.itertuples():
            v_from = self.vertex_idx.get_loc(addr_from)
            v_to = self.vertex_idx.get_loc(addr_to)
            if v_from > v_to:
                v_from, v_to = v_to, v_from
            if (v_from, v_to) in self.weights:
                self.weights[(v_from,v_to)] += 1
            else:
                self.weights[(v_from,v_to)] = 1
                if v_from in self.nexts:
                    self.nexts[v_from].append(v_to)
                else:
                    self.nexts[v_from] = [v_to]
                if v_to in self.nexts:
                    self.nexts[v_to].append(v_from)
                else:
                    self.nexts[v_to] = [v_from]
            self.v_weights[v_from] += 1
            self.v_weights[v_to] += 1
        self.n_edge = len(self.weights)
        print('Edge:', self.n_edge)
        print('Max weight:', max(self.weights.values()), 'Min weight:', min(self.weights.values()))
        print('Max v_weight:', max(self.v_weights), 'Min v_weight:', min(self.v_weights))

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f'{self.n_vertex} {self.n_edge} 011\n')
            for v in range(self.n_vertex):
                f.write(f'{self.v_weights[v]}')
                for v_next in self.nexts.get(v, []):
                    if v < v_next:
                        v_from, v_to = v, v_next
                    else:
                        v_from, v_to = v_next, v
                    weight = self.weights[(v_from, v_to)]
                    f.write(f' {v_next+1} {weight}')
                f.write('\n')

class GroupGraph(Graph):
    def __init__(self, txs, g=7, addr_len=16):
        self.txs = txs
        self.n_vertex = 1<<g
        print('Vertex:', self.n_vertex)
        self.nexts = dict()
        self.weights = dict()
        self.v_weights = np.zeros(shape=self.n_vertex, dtype=int)
        self.shift = addr_len - g
        txs = txs[['from_addr','to_addr']]
        self.addr_group = {}
        for index, addr_from, addr_to in txs.itertuples():
            v_from = addr_from >> self.shift
            v_to = addr_to >> self.shift
            if addr_from not in self.addr_group:
                self.addr_group[addr_from] = v_from
            if addr_to not in self.addr_group:
                self.addr_group[addr_to] = v_to
            if v_from > v_to:
                v_from, v_to = v_to, v_from
            if (v_from, v_to) in self.weights:
                self.weights[(v_from,v_to)] += 1
            else:
                self.weights[(v_from,v_to)] = 1
                if v_from in self.nexts:
                    self.nexts[v_from].append(v_to)
                else:
                    self.nexts[v_from] = [v_to]
                if v_to in self.nexts:
                    self.nexts[v_to].append(v_from)
                else:
                    self.nexts[v_to] = [v_from]
            self.v_weights[v_from] += 1
            self.v_weights[v_to] += 1
        self.n_edge = len(self.weights)
        print('Edge:', self.n_edge)
        print('Max weight:', max(self.weights.values()), 'Min weight:', min(self.weights.values()))
        print('Max v_weight:', max(self.v_weights), 'Min v_weight:', min(self.v_weights))

class PopularGroupGraph(Graph):
    def __init__(self, txs, n_groups):
        self.txs = txs
          
