from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import Graph
from graph.partition import Partition
from strategy.account import StaticAccountAllocate, TableDoubleAccountAllocate
from env.harmony import HarmonySimulator
from env.client import DoubleAddrClient, Client
from exp.log import get_logger

log = get_logger('test_harmony', file_name='logs/test_harmony.log')

def test_harmony(txs, method=['last'], args=None):
    txs = txs[['from','to']]
    n_shards = args.n_shards
    n_blocks = args.n_blocks
    tx_rate = args.tx_rate
    tx_per_block = args.tx_per_block
    block_interval = args.block_interval
    static_allocator = StaticAccountAllocate(n_shards=n_shards)
    
    simulator_args = dict(client=None, allocate=static_allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval, shard_allocation=False)
    
    if 'none' in method:
        log.print('Static allocation:')
        simulator_args['client'] = Client(txs=txs, tx_rate=tx_rate)
        simulator_args['allocate'] = static_allocator
        simulator_args['shard_allocation'] = False
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(None)
        print(simulator.info())

    if 'all' in method:
        log.print('Table updated by all txs partition:')
        client = DoubleAddrClient(txs=txs, tx_rate=tx_rate)
        simulator_args['client'] = client
        table_allocator = TableDoubleAccountAllocate(n_shards=n_shards, fallback=static_allocator)
        simulator_args['allocate'] = table_allocator
        simulator_args['shard_allocation'] = False
        simulator = HarmonySimulator(**simulator_args)
        graph = Graph(txs=client.next(simulator.max_epochs*simulator.epoch_time, peek=True), debug=True).save('./metis/graphs/test_harmony_all.txt')
        account_table = graph.partition(n_shards, debug=True)
        table_allocator.set_account_table(account_table)
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(None)
        log.print(simulator.info())
    
    if 'last' in method:
        log.print('Table updated by last step partition:')
        client = DoubleAddrClient(txs=txs, tx_rate=tx_rate)
        simulator_args['client'] = client
        table_allocator = TableDoubleAccountAllocate(n_shards=n_shards, fallback=static_allocator)
        simulator_args['allocate'] = table_allocator
        simulator_args['shard_allocation'] = False
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        account_table = {}
        for _ in tqdm(range(simulator.max_epochs)):
            done = simulator.step(account_table)
            if done: break
            graph = Graph(simulator.get_block_txs(-simulator.n_blocks)).save('./metis/graphs/test_harmony_last.txt')
            account_table = graph.partition(n_shards)
        print(simulator.info(simulator.n_blocks))
    
    if 'shard' in method:
        log.print('Allocation by shard:')
        client = DoubleAddrClient(txs=txs, tx_rate=tx_rate)
        simulator_args['client'] = client
        table_allocator = TableDoubleAccountAllocate(n_shards=n_shards, fallback=static_allocator)
        simulator_args['allocate'] = table_allocator
        simulator_args['shard_allocation'] = True
        simulator_args['compress'] = args.compress
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        for epoch in tqdm(range(simulator.max_epochs)):
            done = simulator.step(None)
            if done: break
        log.print(simulator.info(simulator.n_blocks))

    if 'pending' in method:
        log.print('Table updated by pending txs partition:')
        client = DoubleAddrClient(txs=txs, tx_rate=tx_rate)
        simulator_args['client'] = client
        table_allocator = TableDoubleAccountAllocate(n_shards=n_shards, fallback=static_allocator)
        simulator_args['allocate'] = table_allocator
        simulator_args['shard_allocation'] = False
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            graph = Graph(simulator.get_pending_txs(forward=False)).save('./metis/graphs/test_harmony_pending.txt')
            account_table = graph.partition(n_shards)
            done = simulator.step(account_table)
            if done: break
        log.print(simulator.info())
    
    simulator.close()

if __name__=='__main__':
    from exp.args import get_default_parser
    parser = get_default_parser(description='test harmony')
    parser.add_argument('--method', type=str, nargs='*', choices=['none', 'all', 'last', 'shard', 'pending'], default=['all'])
    parser.add_argument('--compress', nargs=2, type=int, default=None)
    args = parser.parse_args()
    if args.n_shards==0: args.n_shards = 1 << args.k
    print(args)

    loader = get_default_dataloader()
    _, txs = loader.load_data(start_time=args.start_time, end_time=args.end_time)
    drop_contract_creation_tx(txs)

    test_harmony(txs, method=args.method, args=args)
