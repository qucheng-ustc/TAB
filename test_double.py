from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import Graph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, TableAccountAllocate, DoubleAccountAllocate
from env.eth2 import Eth2v3Simulator
from env.client import DoubleAddrClient

def test_double(txs, method=['last', 'past'], past=[10], args=None):
    n_shards = args.n_shards
    n_blocks = args.n_blocks
    tx_rate = args.tx_rate
    tx_per_block = args.tx_per_block
    block_interval = args.block_interval
    print('Double account addr:')
    graph_path = './metis/graphs/test_double.txt'
    base_allocator = TableAccountAllocate(n_shards=n_shards, fallback=None)
    fallback_allocator = TableAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    allocator = DoubleAccountAllocate(n_shards=n_shards, base=base_allocator, fallback=fallback_allocator)
    txs = txs[['from','to','gas']]
    client = DoubleAddrClient(txs=txs, tx_rate=tx_rate)
    simulator = Eth2v3Simulator(client=client, allocate=allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval)

    if 'none' in method:
        print('Empty table:')
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(None)
        print(simulator.info())

    if 'all' in method:
        print('Table updated by all txs partition:')
        graph = Graph(txs=client.next(simulator.max_epochs*simulator.epoch_time, peek=True), debug=True).save(graph_path)
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
        base_account_table = {}
        fallback_account_table = {}
        for _ in tqdm(range(simulator.max_epochs)):
            done = simulator.step((base_account_table, fallback_account_table))
            if done: break
            graph = Graph(simulator.get_block_txs(-simulator.n_blocks)).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            base_account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
            fallback_account_table = {a[1]:s for a,s in zip(graph.vertex_idx,parts)}
        print(simulator.info(simulator.n_blocks))
    
    if 'past' in method:
        for past_step in past:
            print(f"Table updated by past {past_step} steps partition:")
            simulator.reset(ptx=past_step*simulator.epoch_tx_count)
            for _ in tqdm(range(simulator.max_epochs)):
                graph = Graph(simulator.client.past(past_step*simulator.epoch_time)).save(graph_path)
                parts = Partition(graph_path).partition(n_shards)
                account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
                done = simulator.step(account_table)
                if done: break
            print(simulator.info())
    
    if 'history' in method:
        print("Table updated with all history txs partition:")
        simulator.reset()
        base_account_table = {}
        fallback_account_table = {}
        graph = Graph() # empty graph
        for _ in tqdm(range(simulator.max_epochs)):
            done = simulator.step((base_account_table, fallback_account_table))
            if done: break
            graph = graph.update(simulator.get_block_txs(-simulator.n_blocks)).save(graph_path)
            parts = Partition(graph_path).partition(n_shards)
            base_account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
            fallback_account_table = {a[1]:s for a,s in zip(graph.vertex_idx,parts)}
        print(simulator.info())

if __name__=='__main__':
    from exp.args import get_default_parser
    parser = get_default_parser(description='test double')
    parser.add_argument('--method', type=str, nargs='*', choices=['none', 'all', 'last', 'past', 'current', 'history'], default=['all'])
    parser.add_argument('--past', type=int, nargs='*', default=[20]) # list of number of past steps
    parser.add_argument('--min_size', type=int, default=0)
    args = parser.parse_args()
    if args.n_shards==0: args.n_shards = 1 << args.k
    print(args)

    loader = get_default_dataloader()
    _, txs = loader.load_data(start_time=args.start_time, end_time=args.end_time)
    drop_contract_creation_tx(txs)

    test_double(txs, method=args.method, past=args.past, args=args)
