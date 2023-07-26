class Partition:
    def __init__(self, graph):
        self.graph = graph

    def partition(self, nparts, target_weights=None, allow_imbalance=None, debug=False):
        graph = self.graph
        if isinstance(graph, str):
            import subprocess
            if debug:
                with open(graph, 'r') as f:
                    print('Partition graph:', f.readline().strip())
            options = ['gpmetis', graph, str(nparts)]
            if allow_imbalance is not None:
                options.append(f'-ufactor={allow_imbalance}')
            if target_weights is not None:
                tpwgts_path = f'{graph}.tpwgts.{nparts}'
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
            output_path = f'{graph}.part.{nparts}'
            with open(output_path, 'r') as f:
                parts = [int(line) for line in f]
            return parts
        # else use mymetis
        if graph.n_vertex==0:
            return {}
        # convert graph into metis format
        import mymetis
        xadj = [0]
        adjncy = []
        adjwgt = []
        if graph.v_weight:
            vwgt = []
        i = 0
        for v in range(graph.n_vertex):
            if graph.v_weight:
                vwgt.append(graph.v_weights[v])
            for v_next in graph.nexts[v]:
                weight = graph.weights[(min(v,v_next),max(v,v_next))]
                adjncy.append(v_next)
                adjwgt.append(weight)
                i += 1
            xadj.append(i)
        _, parts = mymetis.partition(xadj=xadj, adjncy=adjncy, vwgt=vwgt, adjwgt=adjwgt, nparts=nparts)
        return parts
