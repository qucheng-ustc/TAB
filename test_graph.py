import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx, convert_tx_addr
from graph.graph import Graph, GroupGraph, PopularGroupGraph, CoarsenGraph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, PartitionAccountAllocate, TableAccountAllocate
from strategy.allocate import GroupAllocateStrategy
from env.eth2 import Eth2v1Simulator, Eth2v1, Eth2v2Simulator
from env.client import Client, PryClient

def test_graph_table(txs, client='normal', simulator='eth2v1', method=['last', 'past'], past=[10], args=None):
    n_shards = args.n_shards
    n_blocks = args.n_blocks
    tx_rate = args.tx_rate
    tx_per_block = args.tx_per_block
    block_interval = args.block_interval
    print('Account graph & Table allocate:')
    graph_path = './metis/graphs/test_graph_table.txt'
    allocator = TableAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    txs = txs[['from','to','gas']]
    if client == 'pry':
        client = PryClient(txs=txs, tx_rate=tx_rate, n_shards=n_shards, account_table=allocator.account_table)
    else:
        client = Client(txs=txs, tx_rate=tx_rate)
    if simulator == 'eth2v2':
        simulator = Eth2v2Simulator(client=client, allocate=allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval)
    else:
        simulator = Eth2v1Simulator(client=client, allocate=allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval)

    if 'none' in method:
        print('Empty table:')
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step({})
        print(simulator.info())

    if 'all' in method:
        print('Table updated by all txs partition:')
        graph = Graph(txs=txs, debug=True).save(graph_path)
        parts = Partition(graph_path).partition(n_shards, debug=True)
        print('Parts:', len(parts))
        account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(account_table)
        print(simulator.info())
    
    if 'current' in method:
        print("Table updated by current partition:")
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            graph = Graph(client.next(simulator.epoch_time, peek=True)).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
            done = simulator.step(account_table)
            if done: break
        print(simulator.info())
    
    if 'last' in method:
        print('Table updated by last step partition:')
        simulator.reset()
        account_table = {}
        for _ in tqdm(range(simulator.max_epochs)):
            done = simulator.step(account_table)
            if done: break
            graph = Graph(simulator.get_block_txs(-simulator.n_blocks)).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
        print(simulator.info())
    
    if 'past' in method:
        for past_step in past:
            print(f"Table updated by past {past_step} steps partition:")
            simulator.reset()
            account_table = {}
            for _ in tqdm(range(simulator.max_epochs)):
                done = simulator.step(account_table)
                if done: break
                graph = Graph(simulator.get_block_txs(-simulator.n_blocks*past_step)).save(graph_path)
                parts = Partition(graph_path).partition(n_shards)
                account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
            print(simulator.info())
    
    if 'history' in method:
        print("Table updated with all history txs partition:")
        simulator.reset()
        account_table = {}
        graph = Graph() # empty graph
        for epoch in tqdm(range(simulator.max_epochs)):
            done = simulator.step(account_table)
            if done: break
            graph = graph.update(simulator.get_block_txs(-simulator.n_blocks)).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
        print(simulator.info())

def test_graph(txs, client='normal', simulator='eth2v1', method=['all', 'last', 'past', 'current', 'history'], past=[100], args=None):
    n_shards = args.n_shards
    n_blocks = args.n_blocks
    tx_rate = args.tx_rate
    tx_per_block = args.tx_per_block
    block_interval = args.block_interval
    print('Account Graph:')
    txs = txs[['from','to','gas']]
    graph_path = './metis/graphs/test_graph.txt'
    allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    if client == 'pry':
        assert(len(method)==1 and method[0]=='none')
        client = PryClient(txs=txs, tx_rate=tx_rate, n_shards=n_shards, account_table={})
    else:
        client = Client(txs=txs, tx_rate=tx_rate)
    if simulator=='eth2v2':
        simulator = Eth2v2Simulator(client=client, allocate=allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval)
    else:
        simulator = Eth2v1Simulator(client=client, allocate=allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval)

    if 'none' in method:
        print('No partition:')
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(([], []))
        print(simulator.info())

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
        for _ in tqdm(range(simulator.max_epochs)):
            graph = Graph(client.next(simulator.epoch_time, peek=True)).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            done = simulator.step((graph.vertex_idx, parts))
            if done: break
        print(simulator.info())
    
    if 'last' in method:
        print("Partition with last step txs:")
        simulator.reset()
        account_list = []
        parts = []
        for _ in tqdm(range(simulator.max_epochs)):
            done = simulator.step((account_list, parts))
            if done: break
            graph = Graph(simulator.epoch_txs).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            account_list = graph.vertex_idx
        print(simulator.info())

    if 'past' in method:
        for past_step in past:
            print(f"Partition with past {past_step} steps txs:")
            simulator.reset(ptx=past_step*simulator.epoch_tx_count)
            for _ in tqdm(range(simulator.max_epochs)):
                graph = Graph(simulator.txs[simulator.client.ptx-past_step*simulator.epoch_tx_count:simulator.client.ptx]).save(graph_path)
                parts = Partition(graph_path).partition(n_shards)
                account_list = graph.vertex_idx
                done = simulator.step((account_list, parts))
                if done: break
            print(simulator.info())

    if 'history' in method:
        print("Partition with all history txs:")
        simulator.reset()
        account_list = []
        parts = []
        graph = None
        for epoch in tqdm(range(simulator.max_epochs)):
            done = simulator.step((account_list, parts))
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
    parser.add_argument('funcs', nargs='*', default=['graph'], choices=['graph', 'group', 'popular', 'coarsen', 'table'])
    parser.add_argument('--method', type=str, nargs='*', choices=['none', 'all', 'last', 'past', 'current', 'history'], default=['all'])
    parser.add_argument('--past', type=int, nargs='*', default=[20]) # list of number of past steps
    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('-g', '--g', type=int, default=10)
    parser.add_argument('--tx_rate', type=int, default=100)
    parser.add_argument('--n_blocks', type=int, default=10) # number of blocks per step
    parser.add_argument('--tx_per_block', type=int, default=200)
    parser.add_argument('--block_interval', type=int, default=15)
    parser.add_argument('--start_time', type=str, default='2021-08-01 00:00:00')
    parser.add_argument('--end_time', type=str, default=None)
    parser.add_argument('--client', type=str, default='normal', choices=['normal', 'pry'])
    parser.add_argument('--simulator', type=str, default='eth2v1', choices=['eth2v1', 'eth2v2'])
    args = parser.parse_args()
    print(args)

    loader = get_default_dataloader()
    _, txs = loader.load_data(start_time=args.start_time, end_time=args.end_time)
    drop_contract_creation_tx(txs)

    k = args.k
    g = args.g
    n_shards = 1 << k
    n_groups = 1 << g
    args.n_shards = n_shards

    func_dict = {
        'graph':lambda:test_graph(txs, client=args.client, simulator=args.simulator, method=args.method, past=args.past, args=args),
        'group':lambda:test_group_graph(txs, k=k, g=g, addr_len=addr_len, tx_rate=args.tx_rate),
        'popular':lambda:test_popular_graph(txs, n_shards=n_shards, n_groups=n_groups, tx_rate=args.tx_rate, method=args.method, past=args.past),
        'coarsen':lambda:test_coarsen_graph(txs, n_shards=n_shards, n_groups=n_groups, tx_rate=args.tx_rate),
        'table':lambda:test_graph_table(txs, client=args.client, simulator=args.simulator, method=args.method, past=args.past, args=args)}

    for func in args.funcs:
        func_dict[func]()
