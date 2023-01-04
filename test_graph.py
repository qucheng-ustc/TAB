from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import Graph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, PartitionAccountAllocate
from env.eth2 import Eth2v1Simulator

if __name__=='__main__':
    n_shards = 64

    loader = get_default_dataloader()
    blocks, txs = loader.load_data(start_time='2021-08-01 00:00:00')
    drop_contract_creation_tx(txs)
    graph = Graph(txs=txs)
    graph_path = './metis/graphs/test_graph.txt'
    graph.save(graph_path)
    partition = Partition(graph_path)
    parts = partition.partition(n_shards)
    print('Parts:', len(parts))

    txs['from_addr'] = txs['from']
    txs['to_addr'] = txs['to']
    txs = txs[['from_addr','to_addr','gas']]
    allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    simulator = Eth2v1Simulator(txs=txs, allocate=allocator, n_shards=n_shards)

    done = simulator.reset()
    while not done:
        done = simulator.step((graph.vertex_idx, parts))
    print(simulator.info())

