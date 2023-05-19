from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import Graph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, TableAccountAllocate, DoubleAccountAllocate
from env.harmony import HarmonySimulator
from env.client import DoubleAddrClient
from exp.log import get_logger

log = get_logger('test_harmony', file_name='logs/test_harmony.log')

def test_harmony(txs, method=['last'], args=None):
    n_shards = args.n_shards
    n_blocks = args.n_blocks
    tx_rate = args.tx_rate
    tx_per_block = args.tx_per_block
    block_interval = args.block_interval
    graph_path = './metis/graphs/test_harmony.txt'
    base_allocator = TableAccountAllocate(n_shards=n_shards, fallback=None)
    fallback_allocator = TableAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    allocator = DoubleAccountAllocate(n_shards=n_shards, base=base_allocator, fallback=fallback_allocator)
    txs = txs[['from','to']]
    client = DoubleAddrClient(txs=txs, tx_rate=tx_rate)
    simulator = HarmonySimulator(client=client, allocate=allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval)

    if 'none' in method:
        log.print('Empty table:')
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(None)
        print(simulator.info())

    if 'all' in method:
        log.print('Table updated by all txs partition:')
        graph = Graph(txs=client.next(simulator.max_epochs*simulator.epoch_time, peek=True), debug=True).save(graph_path)
        parts = Partition(graph_path).partition(n_shards, debug=True)
        log.print('Parts:', len(parts))
        base_account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
        fallback_account_table = {a[1]:s for a,s in zip(graph.vertex_idx,parts)}
        allocator.apply(({base_account_table, fallback_account_table}))
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(None)
        log.print(simulator.info())
    
    if 'last' in method:
        log.print('Table updated by last step partition:')
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
        log.print(simulator.info(simulator.n_blocks))

    if 'pending' in method:
        log.print('Table updated by pending txs partition:')
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            graph = Graph(simulator.get_pending_txs(forward=False), v_weight=vweight).save(graph_path)
            account_table = {}
            if graph.n_edge>0:
                parts = Partition(graph_path).partition(n_shards)
                account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
            done = simulator.step(account_table)
            if done: break
        log.print(simulator.info())
    
    simulator.close()

if __name__=='__main__':
    from exp.args import get_default_parser
    parser = get_default_parser(description='test harmony')
    parser.add_argument('--method', type=str, nargs='*', choices=['none', 'all', 'last', 'pending'], default=['all'])
    args = parser.parse_args()
    if args.n_shards==0: args.n_shards = 1 << args.k
    print(args)

    loader = get_default_dataloader()
    _, txs = loader.load_data(start_time=args.start_time, end_time=args.end_time)
    drop_contract_creation_tx(txs)

    test_harmony(txs, method=args.method, args=args)
