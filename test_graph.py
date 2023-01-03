from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import Graph
from graph.partition import Partition


if __name__=='__main__':
    loader = get_default_dataloader()
    blocks, txs = loader.load_data(start_time='2021-08-01 00:00:00')
    drop_contract_creation_tx(txs)
    graph = Graph(txs=txs)
    graph_path = './metis/graphs/test_graph.txt'
    graph.save(graph_path)
    partition = Partition(graph_path)
    parts = partition.partition(64)
    print('Parts:', parts)

