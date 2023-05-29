import numpy as np
import pandas as pd
import collections
import itertools

# Undirected graph with edge weights and vertex weights(Optional)
class Graph:
    def __init__(self, txs=None, v_weight=True, debug=False):
        self.txs = txs
        self.v_weight = v_weight
        if txs is None or len(txs)==0:
            vertex_idx = pd.Index([],dtype=object)
        else:
            vertex_idx_from = pd.Index(txs['from'].unique())
            vertex_idx_to = pd.Index(txs['to'].unique())
            if debug:
                print('Vertex from:', len(vertex_idx_from), 'Vertex to:', len(vertex_idx_to))
            vertex_idx = vertex_idx_from.union(vertex_idx_to)
        self._init(vertex_idx)
        if debug:
            print('Vertex:', self.n_vertex)
        if self.n_vertex==0:
            return
        txs = txs[['from','to']]
        for index, addr_from, addr_to in txs.itertuples():
            v_from = self.vertex_idx.get_loc(addr_from)
            v_to = self.vertex_idx.get_loc(addr_to)
            self._add_edge(v_from, v_to, 1, self.v_weight)
        if debug:
            print('Edge:', self.n_edge)
            if self.n_edge>0:
                print('weight: max', max(self.weights.values()), ' min', min(self.weights.values()), ' avg', np.average(list(self.weights.values())), ' sum', sum(self.weights.values()))
            if self.v_weight and self.n_vertex>0:
                print('v_weight: max', max(self.v_weights), ' min', min(self.v_weights), ' avg', np.average(self.v_weights), ' sum', sum(self.v_weights))
    
    def _init(self, vertex_idx):
        self.vertex_idx = vertex_idx
        self.n_vertex = len(self.vertex_idx)
        self.n_edge = 0
        self.nexts = dict()
        self.weights = dict()
        if self.v_weight:
            self.v_weights = np.zeros(shape=self.n_vertex, dtype=int)
        
    def _add_edge(self, v_from, v_to, weight=1, v_weight=True):
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
        if v_weight:
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
        if self.v_weight:
            self.v_weights.resize(len(self.vertex_idx))
        txs = txs[['from', 'to']]
        for index, addr_from, addr_to in txs.itertuples():
            v_from = self.vertex_idx.get_loc(addr_from)
            v_to = self.vertex_idx.get_loc(addr_to)
            self._add_edge(v_from, v_to, 1, self.v_weight)
        if debug:
            print('Updated edge:', self.n_edge)
            print('Max weight:', max(self.weights.values()), ' Min weight:', min(self.weights.values()), ' Avg weight:', np.average(list(self.weights.values())))
            if self.v_weight:
                print('Max v_weight:', max(self.v_weights), ' Min v_weight:', min(self.v_weights), 'Avg v_weight:', np.average(self.v_weights))
        return self
    
    # replace/add v_weights to current graph, vertex_idx must be the same
    def set_vweight(self, vertex_idx, v_weights):
        assert((vertex_idx==self.vertex_idx).all())
        self.v_weights = v_weights
        return self
    
    @classmethod
    def load(cls, path):
        graph = object.__new__(cls)
        with open(path, 'r') as f:
            header = f.readline().strip().split(' ')
            n_vertex, n_edge, format = int(header[0]), int(header[1]), header[2]
            v_weight = format=='011'
            nexts = dict()
            weights = dict()
            if v_weight:
                v_weights = np.zeros(shape=n_vertex, dtype=int)
            for v, line in f:
                data = line.strip().split(' ')
                if v_weight:
                    v_weights[v] = int(data[0])
                    data = data[1:]
                for k in range(0,len(data),2):
                    v_next, weight = int(data[k]), int(data[k+1])
                    if v in nexts:
                        nexts[v].append(v_next)
                    else:
                        nexts[v] = [v_next]
                    v_from, v_to = min(v, v_next), max(v, v_next)
                    weights[(v_from, v_to)] = weight
        graph.n_vertex = n_vertex
        graph.n_edge = n_edge
        graph.v_weight = v_weight
        graph.nexts = nexts
        graph.weights = weights
        if v_weight:
            graph.v_weights = v_weights
        return graph

    def save(self, path, v_weight=None):
        if v_weight is None:
            v_weight = self.v_weight
        self.save_path = path
        with open(path, 'w') as f:
            format = '011' if v_weight else '001'
            f.write(f'{self.n_vertex} {self.n_edge} {format}\n')
            for v in range(self.n_vertex):
                space = ''
                if v_weight:
                    f.write(f'{self.v_weights[v]}')
                    space = ' '
                for v_next in self.nexts.get(v, []):
                    if v < v_next:
                        v_from, v_to = v, v_next
                    else:
                        v_from, v_to = v_next, v
                    weight = self.weights[(v_from, v_to)]
                    f.write(f'{space}{v_next+1} {weight}')
                    space = ' '
                f.write('\n')
        return self
    
    def partition(self, nparts, target_weights=None, allow_imbalance=None, save_path=None, debug=False):
        if self.n_edge<=0:
            return {}
        from graph.partition import Partition
        if save_path is None:
            save_path = self.save_path
        parts = Partition(save_path).partition(nparts=nparts, target_weights=target_weights, allow_imbalance=allow_imbalance, debug=debug)
        partition_table = {a:s for a,s in zip(self.vertex_idx,parts)}
        return partition_table

class HotGraph(Graph):
    def __init__(self):
        pass

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
        self.n_edge = 0
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
        self.vertex_idx = pd.Index(self.addr_group.keys())
        if debug:
            print('Edge:', self.n_edge)
            print('Max weight:', max(self.weights.values()), ' Min weight:', min(self.weights.values()), ' Avg weight:', np.average(list(self.weights.values())))
            print('Max v_weight:', max(self.v_weights), ' Min v_weight:', min(self.v_weights), 'Avg v_weight:', np.average(self.v_weights))
            print('Total v_weights:', sum(self.v_weights))
            print("All grouped addrs:", len(self.addr_group))

class CGraph(Graph):
    def _add_vertex_edge(self, v, rv, raw):
        for rv_next in raw.nexts.get(rv, []):
            rv_from, rv_to = min(rv,rv_next), max(rv,rv_next)
            weight = raw.weights.get((rv_from, rv_to), 0)
            v_next = self.rv_map[rv_next]
            if weight>0 and v!=v_next:
                #print('Add edge:', v, v_next, weight)
                self._add_edge(v, v_next, weight, v_weight=False)
                del raw.weights[(rv_from, rv_to)] # delete this edge to avoid adding it twice

    def __init__(self, raw, matched):
        #print('Matched:', matched)
        self.rv_map = {}
        v = -1
        for rv, rvm in enumerate(matched):
            assert(rvm>=0)
            if rv in self.rv_map:
                continue
            if rvm in self.rv_map:
                continue
            v += 1
            self.rv_map[rv] = v
            self.rv_map[rvm] = v
        #print('Rv_map:', self.rv_map)
        self.n_vertex = v+1
        self.n_edge = 0
        self.nexts = dict()
        self.weights = dict()
        self.v_weights = np.zeros(shape=self.n_vertex, dtype=int)
        v_added = -1
        for rv, rvm in enumerate(matched):
            v = self.rv_map[rv]
            if v<=v_added:
                continue
            v_added = v
            if rv == rvm:
                self.v_weights[v] = raw.v_weights[rv]
            else:
                self.v_weights[v] = raw.v_weights[rv] + raw.v_weights[rvm]
                #rv_from, rv_to = min(rv,rvm), max(rv,rvm)
                #if (rv_from, rv_to) in raw.weights:
                    #self.v_weights[v] -= raw.weights[(rv_from, rv_to)]
            #print('rv pair:', rv, rvm, ' v:', v, ' v_weight:', self.v_weights[v])
            self._add_vertex_edge(v, rv, raw)
            if rv == rvm:
                continue
            self._add_vertex_edge(v, rvm, raw)

class CoarsenGraph(Graph):
    @classmethod
    def from_graph(cls, graph, n_vertex, debug=False):
        self = cls.__new__(cls)
        self.raw = graph
        self.coarsen_to(n_vertex, debug=debug)
        return self

    def __init__(self, txs, n_vertex, debug=False):
        self.txs = txs
        if debug:
            print("Coarsen graph -- raw graph:")
        self.raw = Graph(txs, debug=debug)
        self.coarsen_to(n_vertex, debug=debug)

    def coarsen_to(self, n_vertex, debug=False):
        # merge vertexes until n_vertex
        self.target_n_vertex = n_vertex
        self.vertex_idx = self.raw.vertex_idx
        self.rv_map = np.arange(self.raw.n_vertex)
        max_v_weight = 1.5*sum(self.raw.v_weights)/n_vertex # subgraph size constraint
        if debug:
            print('v_weight constraint:', max_v_weight)
        cgraph = self.raw
        level = 0
        while cgraph.n_vertex > n_vertex:
            level += 1
            nmatched = 0
            # match vertexes by its v_weights in ascending order
            sort_v = cgraph.v_weights.argsort()
            matched = np.full(shape=len(sort_v), fill_value=-1, dtype=int)
            i_unmatched = 0
            two_hop = list()
            for i, v in enumerate(sort_v):
                # if already matched, skip
                if matched[v]>=0:
                    continue
                max_v = v
                cnexts = cgraph.nexts.get(v, [])
                # deal with island vertex, match it with an unmatched vertex
                if len(cnexts)==0:
                    i_unmatched = max(i, i_unmatched)+1
                    for i_unmatched in range(i_unmatched, len(sort_v)):
                        v_unmatched = sort_v[i_unmatched]
                        if matched[v_unmatched]==-1:
                            max_v = v_unmatched
                            break
                else:
                    # match vertexes by heavy edge
                    max_weight = -1
                    for v_next in cnexts:
                        if matched[v_next]>=0:
                            continue
                        weight = cgraph.weights[(min(v, v_next), max(v, v_next))]
                        if weight > max_weight and cgraph.v_weights[v]+cgraph.v_weights[v_next]<=max_v_weight:
                            max_weight = weight
                            max_v = v_next
                if max_v != v:
                    matched[v] = max_v
                    matched[max_v] = v
                    nmatched += 1
                    #print('Match:', v, max_v)
                else:
                    two_hop.append(v)
            if debug:
                print("Match level:", level, " Matched:", nmatched, " Two hop:", len(two_hop))
            # perform two hop matching
            unmatched_i = np.full(shape=len(sort_v), fill_value=-1, dtype=int)
            for v in two_hop:
                if matched[v]>=0:
                    continue
                match_v = v
                for v_next in cgraph.nexts.get(v, []):
                    next_nexts = cgraph.nexts.get(v_next, [])
                    for i in range(unmatched_i[v_next]+1, len(next_nexts)):
                        v_next_next = next_nexts[i]
                        if v_next_next == v:
                            continue
                        if matched[v_next_next]>=0:
                            continue
                        if cgraph.v_weights[v] + cgraph.v_weights[v_next_next]>max_v_weight:
                            continue
                        match_v = v_next_next
                        break
                    unmatched_i[v_next] = i
                    if match_v != v:
                        break
                matched[v] = match_v
                matched[match_v] = v
                #print('Two hop match:', v, match_v)
            # match rest vertexes with themselves
            for v in sort_v:
                if matched[v]==-1:
                    matched[v] = v
            # create coarsen graph
            cgraph = CGraph(cgraph, matched)
            for rv, vc in enumerate(self.rv_map):
                self.rv_map[rv] = cgraph.rv_map[vc]
            if debug:
                print("Match level:", level, " Vertex before:", len(sort_v), " Vertex after:", cgraph.n_vertex)
        self.n_vertex = cgraph.n_vertex
        self.nexts = cgraph.nexts
        self.weights = cgraph.weights
        self.v_weights = cgraph.v_weights
        self.n_edge = cgraph.n_edge
        if debug:
            print('Vertex:', self.n_vertex, ' Edge:', self.n_edge)
            if self.n_edge>0:
                print('weight: max', max(self.weights.values()), ' min', min(self.weights.values()), ' avg', np.average(list(self.weights.values())), ' sum', sum(self.weights.values()))
            if self.n_vertex>0:
                print('v_weight: max', max(self.v_weights), ' min', min(self.v_weights), ' avg', np.average(self.v_weights), ' sum', sum(self.v_weights))

class ClusterGraph(Graph):
    def __init__(self, txs, n_cluster, debug=False):
        self.txs = txs
        if debug:
            print("Cluster graph -- raw graph:")
        self.raw = Graph(txs, debug=debug)
        self.cluster(n_cluster, debug=debug)

    def cluster(self, n_cluster, debug=False):
        # sort v_weights in descending order
        sort_v_weights = pd.Series(self.raw.v_weights).sort_values(ascending=False)
        sort_v = sort_v_weights.index
        if debug:
            print('sort_v_weights:', len(sort_v_weights), ' first', sort_v_weights.iloc[0], ' last', sort_v_weights.iloc[-1], ' avg', sort_v_weights.mean(), ' sum', sort_v_weights.sum())
            print('sort_v:', len(sort_v), ' first', sort_v[0], ' last', sort_v[-1], ' min', min(sort_v), ' max', max(sort_v))
        max_v_weight = 1.5*sum(self.raw.v_weights)/n_cluster # cluster size constraint
        
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