from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx, convert_tx_addr
from graph.graph import Graph, GroupGraph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, PartitionAccountAllocate
from strategy.allocate import GroupAllocateStrategy
from env.eth2 import Eth2v1Simulator, Eth2v1

if __name__=='__main__':
    k = 3
    g = 16
    addr_len = 7
    n_shards = 1 << k
    tx_rate = 100
    skip_account_partition = True

    loader = get_default_dataloader()
    blocks, txs = loader.load_data(start_time='2021-08-01 00:00:00')
    drop_contract_creation_tx(txs)
    
    if not skip_account_partition:
        print('Account Graph:')
        graph = Graph(txs=txs)
        graph_path = './metis/graphs/test_graph.txt'
        graph.save(graph_path)
        partition = Partition(graph_path)
        parts = partition.partition(n_shards)
        print('Parts:', len(parts))
        txs['from_addr'] = txs['from']
        txs['to_addr'] = txs['to']
        allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
        simulator = Eth2v1Simulator(txs=txs[['from_addr','to_addr','gas']], allocate=allocator, n_shards=n_shards, tx_rate=tx_rate)

        done = simulator.reset()
        while not done:
            done = simulator.step((graph.vertex_idx, parts))
        print(simulator.info())

    print('Group Graph:')
    convert_tx_addr(txs, addr_len=addr_len)
    group_graph = GroupGraph(txs, g=g, addr_len=addr_len)
    group_graph_path = './metis/graphs/test_group_graph.txt'
    group_graph.save(group_graph_path)
    group_partition = Partition(group_graph_path)
    group_parts = group_partition.partition(n_shards)
    print('Group parts:', len(group_parts))
    group_allocator = GroupAllocateStrategy(k=k, g=g)
    env = Eth2v1(config=dict(txs=txs[['from_addr','to_addr','gas']], allocate=group_allocator, tx_rate=tx_rate))

    done = 0
    env.reset()
    while not done:
        obs, reward, done, _ = env.step(group_parts)
    print(env.info())

    

