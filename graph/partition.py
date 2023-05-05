import metis
import subprocess
import pathlib

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

    def partition(self, nparts, target_weights=None, debug=False):
        if isinstance(self.metis_graph, str):
            options = ['gpmetis', self.metis_graph, str(nparts)]
            if target_weights is not None:
                tpwgts_path = f'{self.metis_graph}.tpwgts.{nparts}'
                with open(tpwgts_path, 'w') as f:
                    for i in range(nparts):
                        f.write(f'{i}={target_weights[i]}\n')
                options.append(f"-tpwgts={tpwgts_path}")
            if not debug:
                options.append("2>&1 >/dev/null")
            cmd = " ".join(options)
            if debug:
                print(cmd)
            ret = subprocess.call(cmd, shell=True)
            output_path = f'{self.metis_graph}.part.{nparts}'
            with open(output_path, 'r') as f:
                parts = [int(line) for line in f]
        else:
            edgecuts, parts = metis.part_graph(self.metis_graph, nparts=nparts)
        return parts

