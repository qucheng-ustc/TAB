from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import Graph


if __name__=='__main__':
    loader = get_default_dataloader()
    blocks, txs = loader.load_data(start_time='2021-08-01 00:00:00')
    drop_contract_creation_tx(txs)
    graph = Graph(txs=txs)
    graph.save('./metis/graphs/test_graph.txt')


