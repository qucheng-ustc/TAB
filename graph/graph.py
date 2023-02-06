import numpy as np
import pandas as pd
import collections
import itertools

class Graph:
    def __init__(self, txs, debug=False):
        self.txs = txs
        vertex_idx_from = pd.Index(txs['from'].unique())
        vertex_idx_to = pd.Index(txs['to'].unique())
        if debug:
            print('Vertex from:', len(vertex_idx_from), 'Vertex to:', len(vertex_idx_to))
        vertex_idx = vertex_idx_from.union(vertex_idx_to)
        self._init(vertex_idx)
        if debug:
            print('Vertex:', self.n_vertex)
        txs = txs[['from','to']]
        for index, addr_from, addr_to in txs.itertuples():
            v_from = self.vertex_idx.get_loc(addr_from)
            v_to = self.vertex_idx.get_loc(addr_to)
            self._add_edge(v_from, v_to)
        if debug:
            print('Edge:', self.n_edge)
            print('Max weight:', max(self.weights.values()), ' Min weight:', min(self.weights.values()), ' Avg weight:', np.average(list(self.weights.values())))
            print('Max v_weight:', max(self.v_weights), ' Min v_weight:', min(self.v_weights), 'Avg v_weight:', np.average(self.v_weights))
    
    def _init(self, vertex_idx):
        self.vertex_idx = vertex_idx
        self.n_vertex = len(self.vertex_idx)
        self.nexts = dict()
        self.weights = dict()
        self.v_weights = np.zeros(shape=self.n_vertex, dtype=int)
        
    def _add_edge(self, v_from, v_to, weight=1):
        if v_from > v_to:
            v_from, v_to = v_to, v_from
        if (v_from, v_to) in self.weights:
            self.weights[(v_from,v_to)] += weight
        else:
            self.weights[(v_from,v_to)] = weight
            if v_from in self.nexts:
                self.nexts[v_from].append(v_to)
            else:
                self.nexts[v_from] = [v_to]
            if v_to in self.nexts:
                self.nexts[v_to].append(v_from)
            else:
                self.nexts[v_to] = [v_from]
        self.v_weights[v_from] += weight
        self.v_weights[v_to] += weight
        self.n_edge = len(self.weights)
    
    def update(self, txs, debug=False):
        vertex_idx_from = pd.Index(txs['from'].unique())
        vertex_idx_to = pd.Index(txs['to'].unique())
        vertex_idx = vertex_idx_from.union(vertex_idx_to)
        new_vertex_idx = vertex_idx.difference(self.vertex_idx)
        self.vertex_idx = self.vertex_idx.append(new_vertex_idx)
        self.n_vertex = len(self.vertex_idx)
        if debug:
            print('Updated vertex:', len(self.vertex_idx), ' New vertex:', len(new_vertex_idx))
        self.v_weights.resize(len(self.vertex_idx))
        txs = txs[['from', 'to']]
        for index, addr_from, addr_to in txs.itertuples():
            v_from = self.vertex_idx.get_loc(addr_from)
            v_to = self.vertex_idx.get_loc(addr_to)
            self._add_edge(v_from, v_to)
        if debug:
            print('Updated edge:', self.n_edge)
            print('Max weight:', max(self.weights.values()), ' Min weight:', min(self.weights.values()), ' Avg weight:', np.average(list(self.weights.values())))
            print('Max v_weight:', max(self.v_weights), ' Min v_weight:', min(self.v_weights), 'Avg v_weight:', np.average(self.v_weights))
        return self

    def save(self, path):
        self.save_path = path
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
        return self

class GroupGraph(Graph):
    def __init__(self, txs, g=7, addr_len=16, debug=False):
        self.txs = txs
        self.n_vertex = 1<<g
        if debug:
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
            self._add_edge(v_from, v_to)
        if debug:
            print('Edge:', self.n_edge)
            print('Max weight:', max(self.weights.values()), ' Min weight:', min(self.weights.values()), ' Avg weight:', np.average(list(self.weights.values())))
            print('Max v_weight:', max(self.v_weights), ' Min v_weight:', min(self.v_weights), 'Avg v_weight:', np.average(self.v_weights))

class PopularGroupGraph(Graph):
    def __init__(self, txs, n_groups, debug=False):
        self.txs = txs
        if debug:
            print("Popular group graph -- raw graph:")
        self.raw = Graph(txs, debug=debug)
        self.n_vertex = n_groups
        # find top n_groups popular addrs
        raw_v_weights = pd.Series(self.raw.v_weights, index=self.raw.vertex_idx)
        topn_weights = raw_v_weights.nlargest(self.n_vertex)
        topn_v = topn_weights.index
        if debug:
            print("Popular:", len(topn_v), " Max weight:", max(topn_weights.values), " Min weight:", min(topn_weights.values), " Avg weight:", np.average(topn_weights.values))
            print("Total weights:", sum(self.raw.v_weights), " Popular weights:", sum(topn_weights.values), " Proportion:", 1.0*sum(topn_weights.values)/sum(self.raw.v_weights))
        # resolve groups, all vertexes directly connected to popular addr
        self.addr_group = {}
        self.addr_weight = {}
        for group_idx, addr in enumerate(topn_v):
            raw_v = self.raw.vertex_idx.get_loc(addr)
            for raw_v_next in self.raw.nexts.get(raw_v, []):
                addr = self.raw.vertex_idx[raw_v_next]
                if addr in topn_v:
                    continue
                weight = self.raw.weights[(min(raw_v, raw_v_next), max(raw_v, raw_v_next))]
                if addr not in self.addr_group:
                    self.addr_group[addr] = group_idx
                    self.addr_weight[addr] = weight
                elif weight > self.addr_weight[addr]:
                    self.addr_group[addr] = group_idx
                    self.addr_weight[addr] = weight
        if debug:
            print("Groups:", len(self.addr_group))
            print("Max weight:", max(self.addr_weight.values()), " Min weight:", min(self.addr_weight.values()), " Avg weight:", np.average(list(self.addr_weight.values())))
            print("Total weight sum:", sum(self.raw.weights.values()), " Group weight sum:", sum(self.addr_weight.values()), " Proportion:", 1.0*sum(self.addr_weight.values())/sum(self.raw.weights.values()))
        # construct group graph
        self.vertex_idx = topn_v
        self.nexts = dict()
        self.weights = dict()
        self.v_weights = np.zeros(shape=self.n_vertex, dtype=int)
        for addr, weight in self.addr_weight.items():
            self.v_weights[self.addr_group[addr]] += weight
        if debug:
            print('Popular Vertex:', self.n_vertex)
            print("Max weight:", max(self.v_weights), " Min weight:", min(self.v_weights), " Avg weight:", np.average(self.v_weights))
        for addr, v in self.addr_group.items():
            raw_v = self.raw.vertex_idx.get_loc(addr)
            for raw_v_next in self.raw.nexts.get(raw_v, []):
                addr = self.raw.vertex_idx[raw_v_next]
                weight = self.raw.weights[(min(raw_v, raw_v_next), max(raw_v, raw_v_next))]
                if addr not in self.addr_group:
                    continue
                v_next = self.addr_group[addr]
                if v == v_next:
                    continue
                self._add_edge(v, v_next, weight)
        self.addr_group.update({v:i for i,v in enumerate(topn_v.values)})
        if debug:
            print('Edge:', self.n_edge)
            print('Max weight:', max(self.weights.values()), ' Min weight:', min(self.weights.values()), ' Avg weight:', np.average(list(self.weights.values())))
            print('Max v_weight:', max(self.v_weights), ' Min v_weight:', min(self.v_weights), 'Avg v_weight:', np.average(self.v_weights))
            print('Total v_weights:', sum(self.v_weights))
            print("All grouped addrs:", len(self.addr_group))

class CGraph(Graph):
    def __init__(self, raw, v_match):
        self.n_vertex = len(v_match)
        self.nexts = dict()
        self.weights = dict()
        self.v_weights = np.zeros(shape=self.n_vertex, dtype=int)
        self.rv_map = {}
        for i, (rvp, rvq) in enumerate(v_match.items()):
            self.rv_map[rvp] = i
            self.rv_map[rvq] = i
        for rvp, rvq in v_match.items():
            v = self.rv_map[rvp]
            rv_from, rv_to = min(rvp, rvq), max(rvp,rvq)
            if (rv_from, rv_to) in raw.weights:
                self.v_weights[v] += raw.weights[(rv_from, rv_to)]
            for rv_next in raw.nexts.get(rv_from, []):
                weight = raw.weights[min(rv_from,rv_next),max(rv_from,rv_next)]
                self._add_edge(v, self.rv_map[rv_next], weight)
            for rv_next in raw.nexts.get(rv_to, []):
                weight = raw.weights[min(rv_to,rv_next),max(rv_to,rv_next)]
                self._add_edge(v, self.rv_map[rv_next], weight)

class CoarsenGraph(Graph):
    def __init__(self, txs, n_vertex, debug=False):
        self.txs = txs
        if debug:
            print("Coarsen graph -- raw graph:")
        self.raw = Graph(txs, debug=debug)
        # merge vertexes until n_vertex
        self.vertex_idx = self.raw.vertex_idx
        self.rv_map = np.arange(self.raw.n_vertex)
        cgraph = self.raw
        level = 0
        while cgraph.n_vertex > n_vertex:
            level += 1
            # match vertexes by its v_weights in ascending order
            v_match = {}
            matched = set()
            sort_v = cgraph.v_weights.argsort()
            i_unmatched = 0
            for i, v in enumerate(sort_v):
                # if already matched, skip
                if v in matched:
                    continue
                max_v = v # default match with itself
                cnexts = cgraph.nexts.get(v, [])
                # deal with island vertex, match it with an unmatched vertex
                if len(cnexts)==0:
                    i_unmatched = max(i, i_unmatched)+1
                    for i_unmatched in range(i_unmatched, len(sort_v)):
                        v_unmatched = sort_v[i_unmatched]
                        if v_unmatched not in matched:
                            max_v = v_unmatched
                            break
                else:
                    # match vertexes by heavy edge, match with itself if no match found
                    max_weight = -1
                    for v_next in cnexts:
                        if v_next in matched:
                            continue
                        weight = cgraph.weights[(min(v, v_next), max(v, v_next))]
                        if weight > max_weight:
                            max_weight = weight
                            max_v = v_next
                v_match[v] = max_v
                matched.add(v)
                matched.add(max_v)
            # create coarsen graph
            cgraph = CGraph(cgraph, v_match)
            for rv, vc in enumerate(self.rv_map):
                self.rv_map[rv] = cgraph.rv_map[vc]
            if debug:
                print("Match level:", level, " Vertex:", cgraph.n_vertex)
        self.n_vertex = cgraph.n_vertex
        self.nexts = cgraph.nexts
        self.weights = cgraph.weights
        self.v_weights = cgraph.v_weights
        self.n_edge = cgraph.n_edge
        if debug:
            print('Vertex:', self.n_vertex, ' Edge:', self.n_edge)
            print('Max weight:', max(self.weights.values()), ' Min weight:', min(self.weights.values()), ' Avg weight:', np.average(list(self.weights.values())))
            print('Max v_weight:', max(self.v_weights), ' Min v_weight:', min(self.v_weights), 'Avg v_weight:', np.average(self.v_weights))
            print('Total v_weights:', sum(self.v_weights))
