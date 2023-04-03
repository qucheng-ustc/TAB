import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, find
import tqdm

from .graph import Graph

def spm_col_max(spm):
    csc = spm.tocsc()
    cmax = np.zeros(shape=csc.shape[1])
    for i in range(csc.shape[1]):
        row_ind, col_ind, data = find(csc.getcol(i))
        if len(data)>0:
            cmax[i] = np.max(data)
    return cmax

def spm_col_min(spm):
    csc = spm.tocsc()
    cmin = np.zeros(shape=csc.shape[1])
    for i in range(csc.shape[1]):
        row_ind, col_ind, data = find(csc.getcol(i))
        if len(data)>0:
            cmin[i] = np.min(data)
    return cmin

class GraphStack:
    def __init__(self, block_txs, debug=False):
        self.block_txs = block_txs
        self.blocks = block_txs['block'].unique()
        self.size = len(self.blocks)
        if debug:
            print('Blocks:', min(self.blocks),'~', max(self.blocks), ' Size:', self.size)
        vertex_idx_from = pd.Index(block_txs['from'].unique())
        vertex_idx_to = pd.Index(block_txs['to'].unique())
        self.vertex_idx = vertex_idx_from.union(vertex_idx_to)
        self.n_vertex = len(self.vertex_idx)
        if debug:
            print('Vertex from:', len(vertex_idx_from), 'Vertex to:', len(vertex_idx_to))
            print('Vertex:', self.n_vertex)
        self.n_edge = 0
        self.new_edges = [[] for _ in range(self.size)]
        self.new_vertexes = [[] for _ in range(self.size)]
        self.layer_edges = [0]*self.size
        self.nexts = dict()
        self.weights = dict()
        iter_groups = enumerate(block_txs.groupby('block'))
        if debug:
            iter_groups = tqdm.tqdm(iter_groups, desc='Layer')
        for layer, (block, txs) in iter_groups:
            for index, addr_from, addr_to in txs[['from','to']].itertuples():
                v_from = self.vertex_idx.get_loc(addr_from)
                v_to = self.vertex_idx.get_loc(addr_to)
                if v_from > v_to:
                    v_from, v_to = v_to, v_from
                if (v_from, v_to) in self.weights:
                    continue
                self.weights[(v_from,v_to)] = len(self.weights)
                if v_from in self.nexts:
                    self.nexts[v_from].append(v_to)
                else:
                    self.nexts[v_from] = [v_to]
                    self.new_vertexes[layer].append(v_from)
                if v_to in self.nexts:
                    self.nexts[v_to].append(v_from)
                else:
                    self.nexts[v_to] = [v_from]
                    self.new_vertexes[layer].append(v_to)
                self.n_edge += 1
                self.new_edges[layer].append(self.weights[(v_from,v_to)])
        self.weight_index = pd.Index(list(self.weights.keys()))
        self.weight_matrix = lil_matrix((len(self.weights), self.size), dtype=int)
        self.vweight_matrix = lil_matrix((self.n_vertex, self.size), dtype=int)
        iter_groups = enumerate(block_txs.groupby('block'))
        if debug:
            iter_groups = tqdm.tqdm(iter_groups, desc='Weight')
        for layer, (block, txs) in iter_groups:
            for index, addr_from, addr_to in txs[['from','to']].itertuples():
                v_from = self.vertex_idx.get_loc(addr_from)
                v_to = self.vertex_idx.get_loc(addr_to)
                if v_from > v_to:
                    v_from, v_to = v_to, v_from
                weight_id = self.weight_index.get_loc((v_from, v_to))
                if self.weight_matrix[weight_id,layer]==0:
                    self.layer_edges[layer] += 1
                self.weight_matrix[weight_id,layer] += 1
                self.vweight_matrix[v_from,layer] += 1
                self.vweight_matrix[v_to,layer] += 1
        if debug:
            new_vertex_num = [len(v) for v in self.new_vertexes]
            print('Vertex:', self.n_vertex, ', new vertexes:', new_vertex_num)
            new_edge_num = [len(e) for e in self.new_edges]
            print('Edge:', self.n_edge, ', layer edges:', self.layer_edges, ', new edges:', new_edge_num, 'sum', sum(new_edge_num))
            new_vweight_matrix = lil_matrix((self.n_vertex, self.size), dtype=int)
            for layer in range(self.size):
                for v in self.new_vertexes[layer]:
                    new_vweight_matrix[v,layer] = self.vweight_matrix[v,layer]
            vweight_layer_sum = self.vweight_matrix.sum(axis=0)
            new_vweight_layer_sum = new_vweight_matrix.sum(axis=0)
            vweight_layer_mean = vweight_layer_sum/self.vweight_matrix.getnnz(axis=0)
            new_vweight_layer_mean = new_vweight_layer_sum/new_vweight_matrix.getnnz(axis=0)
            print('vweights sum:', vweight_layer_sum, 'new vweights sum:', new_vweight_layer_sum, 'prop:', new_vweight_layer_sum/vweight_layer_sum)
            print('vweights mean:', vweight_layer_mean, 'new vweights mean:', new_vweight_layer_mean)
            new_weight_matrix = lil_matrix((len(self.weights), self.size), dtype=int)
            for layer in range(self.size):
                for ei in self.new_edges[layer]:
                    new_weight_matrix[ei,layer] = self.weight_matrix[ei,layer]
            weight_layer_sum = self.weight_matrix.sum(axis=0)
            new_weight_layer_sum = new_weight_matrix.sum(axis=0)
            print('weights sum:', weight_layer_sum, 'new weights sum:', new_weight_layer_sum, 'prop:', new_weight_layer_sum/weight_layer_sum)
            weight_layer_mean = weight_layer_sum/self.weight_matrix.getnnz(axis=0)
            new_weight_layer_mean = new_weight_layer_sum/new_weight_matrix.getnnz(axis=0)
            print('weights mean:', weight_layer_mean, 'new weights mean:', new_weight_layer_mean)
            weight_layer_max = spm_col_max(self.weight_matrix)
            new_weight_layer_max = spm_col_max(new_weight_matrix)
            print('weights max:', weight_layer_max, 'new weights max:', new_weight_layer_max)
            weight_layer_min = spm_col_min(self.weight_matrix)
            new_weight_layer_min = spm_col_min(new_weight_matrix)
            print('weights min:', weight_layer_min, 'new weights min:', new_weight_layer_min)
    
    def get_weight_matrix(self, start=0, stop=None):
        return self.weight_matrix[:,start:stop]
    def get_vweight_matrix(self, start=0, stop=None):
        return self.vweight_matrix[:,start:stop]

class WeightGraph(Graph):
    def __init__(self, vertex_idx, weight_index, weight_array, vweight_array=None, vweight_weight=0.):
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
        # update v_weights with external vweight_array
        if vweight_array is not None:
            self.v_weights = (1.-vweight_weight)*self.v_weights + vweight_weight*vweight_array
        
