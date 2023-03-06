import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx, convert_tx_addr
from graph.graph import Graph, GroupGraph, PopularGroupGraph, CoarsenGraph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, PartitionAccountAllocate
from strategy.allocate import GroupAllocateStrategy
from env.eth2 import Eth2v1Simulator, Eth2v1

def test_graph(txs, n_shards, tx_rate, method=['all', 'last', 'past', 'current', 'history'], past=[100], n_blocks=10):
    print('Account Graph:')
    graph_path = './metis/graphs/test_graph.txt'
    allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    txs = txs[['from','to','gas']]
    simulator = Eth2v1Simulator(txs=txs, allocate=allocator, n_shards=n_shards, tx_rate=tx_rate, n_blocks=n_blocks)
    
    if 'all' in method:
        print("Partition with all txs:")
        graph = Graph(txs=txs, debug=True).save(graph_path)
        parts = Partition(graph_path).partition(n_shards, debug=True)
        print('Parts:', len(parts))
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step((graph.vertex_idx, parts))
        print(simulator.info())
    
    if 'current' in method:
        print("Partition with current step txs:")
        simulator.reset()
        account_list = []
        parts = []
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step((account_list, parts))
            graph = Graph(simulator.txs[simulator.ptx:min(len(simulator.txs),simulator.ptx+simulator.epoch_tx_count)]).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_list = graph.vertex_idx
        print(simulator.info())
    
    if 'last' in method:
        print("Partition with last step txs:")
        simulator.reset()
        account_list = []
        parts = []
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step((account_list, parts))
            graph = Graph(simulator.epoch_txs).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_list = graph.vertex_idx
        print(simulator.info())

    if 'past' in method:
        for past_step in past:
            print(f"Partition with past {past_step} steps txs:")
            simulator.reset(ptx=past_step*simulator.epoch_tx_count)
            account_list = []
            parts = []
            for _ in tqdm(range(simulator.max_epochs)):
                simulator.step((account_list, parts))
                graph = Graph(simulator.txs[simulator.ptx-past_step*simulator.epoch_tx_count:simulator.ptx]).save(graph_path)
                parts = Partition(graph_path).partition(n_shards)
                account_list = graph.vertex_idx
            print(simulator.info())

    if 'history' in method:
        print("Partition with all history txs:")
        simulator.reset()
        account_list = []
        parts = []
        graph = None
        for epoch in tqdm(range(simulator.max_epochs)):
            simulator.step((account_list, parts))
            if graph is None:
                graph = Graph(simulator.epoch_txs).save(graph_path)
            else:
                graph = graph.update(simulator.epoch_txs).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_list = graph.vertex_idx
        print(simulator.info())

def test_group_graph(txs, k, g, addr_len, tx_rate, method=['all']):
    print('Group Graph:')
    graph_path = './metis/graphs/test_group_graph.txt'
    convert_tx_addr(txs, addr_len=addr_len)
    group_allocator = GroupAllocateStrategy(k=k, g=g)
    env = Eth2v1(config=dict(txs=txs[['from_addr','to_addr','gas']], allocate=group_allocator, tx_rate=tx_rate))

    if 'all' in method:
        print("Partition with all txs:")
        graph = GroupGraph(txs, g=g, addr_len=addr_len, debug=True).save(graph_path)
        parts = Partition(graph_path).partition(n_shards, debug=True)
        print('Group parts:', len(parts))
        env.reset()
        for _ in tqdm(range(env.simulator.max_epochs)):
            obs, reward, done, _ = env.step(parts)
        print(env.info())

    if 'last' in method:
        print("Partition with last step txs:")
        env.reset()
        parts = GroupAllocateStrategy(k=k, g=g).group_table
        for _ in tqdm(range(env.simulator.max_epochs)):
            obs, reward, done, _ = env.step(parts)
            graph = GroupGraph(env.simulator.epoch_txs, g=g, addr_len=addr_len).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
        print(env.info())

def test_popular_graph(txs, n_shards, n_groups, tx_rate, method=['all', 'last', 'current', 'past'], past=[100]):
    print('Popular Group Graph:')
    graph_path = './metis/graphs/test_popular_graph.txt'
    allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    simulator = Eth2v1Simulator(txs=txs[['from','to','gas']], allocate=allocator, n_shards=n_shards, tx_rate=tx_rate)

    if 'all' in method:
        print("Partition with all txs:")
        graph = PopularGroupGraph(txs, n_groups, debug=True).save(graph_path)
        parts = Partition(graph_path).partition(n_shards, debug=True)
        print('Parts:', len(parts))
        account_list = graph.vertex_idx
        account_parts = [parts[graph.addr_group[addr]] for addr in account_list]
        print("Account list:", len(account_list))
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step((account_list, account_parts))
        print(simulator.info())
    
    if 'current' in method:
        print("Partition with current step txs:")
        simulator.reset()
        account_list = []
        account_parts = []
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step((account_list, account_parts))
            graph = PopularGroupGraph(simulator.txs[simulator.ptx:min(len(simulator.txs),simulator.ptx+simulator.epoch_tx_count)], n_groups).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_list = graph.vertex_idx
            account_parts = [parts[graph.addr_group[addr]] for addr in account_list]
        print(simulator.info())

    if 'last' in method:
        print("Partition with last step txs:")
        simulator.reset()
        account_list = []
        account_parts = []
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step((account_list, account_parts))
            graph = PopularGroupGraph(simulator.epoch_txs, n_groups).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_list = graph.vertex_idx
            account_parts = [parts[graph.addr_group[addr]] for addr in account_list]
        print(simulator.info())

    if 'past' in method:
        for past_step in past:
            print(f"Partition with past {past_step} steps txs:")
            simulator.reset()
            account_list = []
            account_parts = []
            for _ in tqdm(range(simulator.max_epochs)):
                simulator.step((account_list, account_parts))
                graph = PopularGroupGraph(simulator.txs[max(0,simulator.ptx-past_step*simulator.epoch_tx_count):simulator.ptx], n_groups).save(graph_path)
                parts = Partition(graph_path).partition(n_shards)
                account_list = graph.vertex_idx
                account_parts = [parts[graph.addr_group[addr]] for addr in account_list]
            print(simulator.info())

def test_coarsen_graph(txs, n_shards, n_groups, tx_rate, method=['all']):
    print('Coarsen Graph:')
    graph_path = './metis/graphs/test_coarsen_graph.txt'
    allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    simulator = Eth2v1Simulator(txs=txs[['from','to','gas']], allocate=allocator, n_shards=n_shards, tx_rate=tx_rate)

    if 'all' in method:
        print("Partition with all txs:")
        graph = CoarsenGraph.from_graph(Graph(txs=txs, debug=True), n_groups, debug=True).save(graph_path)
        parts = Partition(graph_path).partition(n_shards, debug=True)
        print('Parts:', len(parts))
        account_list = graph.vertex_idx
        account_parts = [parts[graph.rv_map[i]] for i in range(len(account_list))]
        print("Account list:", len(account_list))
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step((account_list, account_parts))
        print(simulator.info())

if __name__=='__main__':
    addr_len = 16

    import argparse
    parser = argparse.ArgumentParser(description='test graph')
    parser.add_argument('funcs', nargs='*', default=['graph', 'group', 'popular', 'coarsen'])
    parser.add_argument('--method', type=str, nargs='*', choices=['all', 'last', 'past', 'current', 'history'], default=['all'])
    parser.add_argument('--past', type=int, nargs='*', default=[20, 40, 60, 80, 100]) # number of past steps
    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('-g', '--g', type=int, default=10)
    parser.add_argument('--tx_rate', type=int, default=100)
    parser.add_argument('--n_blocks', type=int, default=10) # number of blocks per step
    args = parser.parse_args()
    print(args)

    loader = get_default_dataloader()
    _, txs = loader.load_data(start_time='2021-08-01 00:00:00')
    drop_contract_creation_tx(txs)

    k = args.k
    g = args.g
    n_shards = 1 << k
    n_groups = 1 << g
    tx_rate = args.tx_rate

    func_dict = {'graph':lambda:test_graph(txs, n_shards=n_shards, tx_rate=tx_rate, method=args.method, past=args.past, n_blocks=args.n_blocks),
                'group':lambda:test_group_graph(txs, k=k, g=g, addr_len=addr_len, tx_rate=tx_rate),
                'popular':lambda:test_popular_graph(txs, n_shards=n_shards, n_groups=n_groups, tx_rate=tx_rate, method=args.method, past=args.past),
                'coarsen':lambda:test_coarsen_graph(txs, n_shards=n_shards, n_groups=n_groups, tx_rate=tx_rate)}

    for func in args.funcs:
        func_dict[func]()
