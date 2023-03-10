import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
import tqdm

from .graph import Graph

class GraphStack:
    def __init__(self, block_txs, debug=False):
        self.block_txs = block_txs
        self.blocks = block_txs['block'].unique()
        self.size = len(self.blocks)
        if debug:
            print('Blocks:', self.blocks, ' Size:', self.size)
        vertex_idx_from = pd.Index(block_txs['from'].unique())
        vertex_idx_to = pd.Index(block_txs['to'].unique())
        self.vertex_idx = vertex_idx_from.union(vertex_idx_to)
        self.n_vertex = len(self.vertex_idx)
        if debug:
            print('Vertex from:', len(vertex_idx_from), 'Vertex to:', len(vertex_idx_to))
            print('Vertex:', self.n_vertex)
        self.n_edge = 0
        self.new_edges = [0]*self.size
        self.layer_edges = [0]*self.size
        self.nexts = dict()
        self.weights = dict()
        iter_groups = enumerate(block_txs.groupby('block'))
        if debug:
            iter_groups = tqdm.tqdm(iter_groups, desc='Layer')
        for l, (block, txs) in iter_groups:
            for index, addr_from, addr_to in txs[['from','to']].itertuples():
                v_from = self.vertex_idx.get_loc(addr_from)
                v_to = self.vertex_idx.get_loc(addr_to)
                self._add_edge(l, v_from, v_to)
        self.weight_index = pd.Index(list(self.weights.keys()))
        if debug:
            print('Edge:', self.n_edge, ', layer edges:', self.layer_edges, ', new edges:', self.new_edges, 'sum', sum(self.new_edges))
    
    def _add_edge(self, layer, v_from, v_to, weight=1):
        if v_from > v_to:
            v_from, v_to = v_to, v_from
        if (v_from, v_to) not in self.weights:
            self.weights[(v_from,v_to)] = lil_matrix((1, self.size), dtype=int)
            if v_from in self.nexts:
                self.nexts[v_from].append(v_to)
            else:
                self.nexts[v_from] = [v_to]
            if v_to in self.nexts:
                self.nexts[v_to].append(v_from)
            else:
                self.nexts[v_to] = [v_from]
            self.n_edge += 1
            self.new_edges[layer] += 1
        if self.weights[(v_from,v_to)][0,layer]==0:
            self.layer_edges[layer] += 1
        self.weights[(v_from,v_to)][0,layer] += weight
    
    def get_weight_matrix(self, start=0, stop=None):
        if stop is None:
            stop = self.size
        self.weight_matrix = lil_matrix((len(self.weights), stop - start), dtype=int)
        for k in self.weights:
            self.weight_matrix[self.weight_index.get_loc(k),:] = self.weights[k][0][start:stop]
        return self.weight_matrix

class WeightGraph(Graph):
    def __init__(self, vertex_idx, weight_index, weight_array):
        # remove zero weights
        vertexes = []
        v_map = {}
        for i, weight in enumerate(weight_array):
            if weight<=0: continue
            v_from, v_to = weight_index[i]
            if v_from not in v_map:
                v_map[v_from] = len(vertexes)
                vertexes.append(vertex_idx[v_from])
            if v_to not in v_map:
                v_map[v_to] = len(vertexes)
                vertexes.append(vertex_idx[v_to])
        self.vertex_idx = pd.Index(vertexes)
        self.n_vertex = len(self.vertex_idx)
        self.weights = dict()
        self.nexts = dict()
        self.v_weights = np.zeros(shape=self.n_vertex, dtype=weight_array.dtype)
        self.n_edge = 0
        for i, weight in enumerate(weight_array):
            if weight<=0: continue
            v_from, v_to = weight_index[i]
            v_from, v_to = v_map[v_from], v_map[v_to]
            self._add_edge(v_from, v_to, weight, v_weight=True)
