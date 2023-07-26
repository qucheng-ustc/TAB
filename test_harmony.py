import os
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from graph.graph import Graph
from strategy.account import StaticAccountAllocate, TableDoubleAccountAllocate, TableAccountAllocate
from env.harmony import HarmonySimulator, Overhead
from env.client import DoubleAddrClient, Client
import exp

import numpy as np
np.random.seed(0)

log = exp.log.get_logger('test_harmony', file_name='logs/test_harmony.log')

def test_harmony(txs, method='last', args=None):
    n_shards = args.n_shards
    n_blocks = args.n_blocks
    tx_rate = args.tx_rate
    tx_per_block = args.tx_per_block
    block_interval = args.block_interval
    static_allocator = StaticAccountAllocate(n_shards=n_shards)

    overhead = None
    if args.overhead:
        overhead = Overhead()
    
    simulator_args = dict(client=None, allocate=static_allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval, shard_allocation=False, 
                          pmatch=args.pmatch, compress=args.compress, overhead=overhead, save_path=args.save_path)

    def get_client():
        if args.double_addr is True:
            return DoubleAddrClient(txs=txs, tx_rate=tx_rate)
        else:
            return Client(txs=txs, tx_rate=tx_rate)
    def get_allocator(method='table'):
        if method == 'none':
            if args.double_addr is True:
                raise NotImplementedError
            else:
                return static_allocator
        if args.double_addr is True:
            return TableDoubleAccountAllocate(n_shards=n_shards, fallback=static_allocator)
        else:
            return TableAccountAllocate(n_shards=n_shards, fallback=static_allocator)
    
    if 'none' == method:
        log.print('Static allocation:')
        simulator_args['client'] = get_client()
        simulator_args['allocate'] = get_allocator('none')
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(None)

    if 'all' == method:
        log.print('Table updated by all txs partition:')
        client = get_client()
        simulator_args['client'] = get_client()
        table_allocator = get_allocator()
        simulator_args['allocate'] = table_allocator
        simulator = HarmonySimulator(**simulator_args)
        graph = Graph(txs=client.next(simulator.max_epochs*simulator.epoch_time, peek=True), debug=True).save('./metis/graphs/test_harmony_all.txt')
        account_table = graph.partition(n_shards, debug=True)
        table_allocator.set_account_table(account_table)
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            simulator.step(None)
    
    if 'last' == method:
        log.print('Table updated by last step partition:')
        simulator_args['client'] = get_client()
        simulator_args['allocate'] = get_allocator()
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        account_table = {}
        for _ in tqdm(range(simulator.max_epochs)):
            done = simulator.step(account_table)
            if done: break
            graph = Graph(simulator.get_block_txs(-simulator.n_blocks)).save('./metis/graphs/test_harmony_last.txt')
            account_table = graph.partition(n_shards)

    if 'pending' == method:
        log.print('Table updated by pending txs partition:')
        simulator_args['client'] = get_client()
        simulator_args['allocate'] = get_allocator()
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        for _ in tqdm(range(simulator.max_epochs)):
            graph = Graph(simulator.get_pending_txs(forward=False)).save('./metis/graphs/test_harmony_pending.txt')
            account_table = graph.partition(n_shards)
            done = simulator.step(account_table)
            if done: break

    if 'shard' == method:
        log.print('Allocation by shard:')
        simulator_args['client'] = get_client()
        simulator_args['allocate'] = get_allocator()
        simulator_args['shard_allocation'] = True
        simulator = HarmonySimulator(**simulator_args)
        simulator.reset()
        for epoch in tqdm(range(simulator.max_epochs)):
            done = simulator.step(None)
            if done: break

    simulator.close()
    info = simulator.info(n_blocks)
    log.print(info)
    return info

if __name__=='__main__':
    parser = exp.args.get_default_parser(description='test harmony')
    parser.add_argument('--method', type=str, choices=['none', 'all', 'last', 'shard', 'pending'], default='shard')
    parser.add_argument('--double_addr', action="store_true")
    parser.add_argument('--compress', nargs="*", type=int, default=None)
    parser.add_argument('--pmatch', action="store_true")
    parser.add_argument('--overhead', action="store_true")
    parser.add_argument('--save_path', type=str, default='/tmp/harmony')
    args = parser.parse_args()
    if args.n_shards==0: args.n_shards = 1 << args.k
    log.print(args)

    os.makedirs(args.save_path, exist_ok=True)
    recorder = exp.recorder.Recorder('records/test_harmony', params=vars(args))

    loader = get_default_dataloader()
    _, txs = loader.load_data(start_time=args.start_time, end_time=args.end_time)
    loader.close()
    drop_contract_creation_tx(txs)
    txs = txs[['from','to']]

    info = test_harmony(txs, method=args.method, args=args)
    recorder.add("info", info)
    recorder.save()
