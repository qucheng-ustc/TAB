import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

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
        self.n_edge = [0]*self.size
        self.nexts = dict()
        self.weights = dict()

        for l, (block, txs) in enumerate(block_txs.groupby('block')):
            if debug:
                print(f'Layer {l}:', 'block', block, 'txs', len(txs))
            for index, addr_from, addr_to in txs[['from','to']].itertuples():
                v_from = self.vertex_idx.get_loc(addr_from)
                v_to = self.vertex_idx.get_loc(addr_to)
                self._add_edge(l, v_from, v_to)
        if debug:
            print('Edge:', self.n_edge, ' sum', sum(self.n_edge))
    
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
            self.n_edge[layer] += 1
        self.weights[(v_from,v_to)][0,layer] += weight
