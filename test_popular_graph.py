import pandas as pd
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import PopularGroupGraph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, PartitionAccountAllocate
from env.eth2 import Eth2v1Simulator

if __name__=='__main__':
    k = 3
    g = 7
    n_groups = 1 << g
    n_shards = 1 << k
    tx_rate = 100

    loader = get_default_dataloader()
    blocks, txs = loader.load_data(start_time='2021-08-01 00:00:00')
    drop_contract_creation_tx(txs)
    
    print('Popular Group Graph:')
    graph = PopularGroupGraph(txs, n_groups)
    graph_path = './metis/graphs/test_popular_graph.txt'
    graph.save(graph_path)
    partition = Partition(graph_path)
    parts = partition.partition(n_shards)
    print('Parts:', len(parts))
    txs['from_addr'] = txs['from']
    txs['to_addr'] = txs['to']
    allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    simulator = Eth2v1Simulator(txs=txs[['from_addr','to_addr','gas']], allocate=allocator, n_shards=n_shards, tx_rate=tx_rate)
    account_list = pd.Index(graph.addr_group.keys())
    account_parts = [parts[graph.addr_group[addr]] for addr in account_list]
    print("Account list:", len(account_list))
    done = simulator.reset()
    while not done:
        done = simulator.step((account_list, account_parts))
    print(simulator.info())
