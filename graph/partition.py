import metis
import os

class Partition:
    def __init__(self, graph):
        if isinstance(graph, str):
            self.metis_graph = graph
        else:
            adjlist = [[]]*graph.n_vertex
            for v in range(graph.n_vertex):
                for v_next in graph.nexts.get(v,[]):
                    v1, v2 = min(v,v_next),max(v,v_next)
                    weight = graph.weights[(v1,v2)]
                    adjlist[v].append((v_next,weight))
            v_weights = graph.v_weights
            self.metis_graph = metis.adjlist_to_metis(adjlist, nodew=v_weights)


    def partition(self, nparts):
        if isinstance(self.metis_graph, str):
            cmd = f'gpmetis {self.metis_graph} {nparts}'
            print(cmd)
            os.system(cmd)
            output_path = f'{self.metis_graph}.part.{nparts}'
            print(output_path)
            with open(output_path, 'r') as f:
                parts = [int(line) for line in f]
        else:
            edgecuts, parts = metis.part_graph(self.metis_graph, nparts=nparts)
        return parts
