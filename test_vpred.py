import numpy as np
import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from strategy.account import StaticAccountAllocate, TableAccountAllocate, DoubleAccountAllocate
from prediction.account import AverageModel, MLPModel
from prediction.metrics import mean_squared_error
from graph.stack import Graph, GraphStack, WeightGraph
from graph.partition import Partition
from env.eth2 import Eth2v3Simulator
from env.client import Client, DoubleAddrClient
from exp.log import get_logger

log = get_logger('test_vpred', file_name='logs/test_vpred.log')

def simulate(txs, model, window, args):
    n_shards = args.n_shards
    n_blocks = args.n_blocks
    tx_rate = args.tx_rate
    tx_per_block = args.tx_per_block
    block_interval = args.block_interval
    log.print("Double account table updated with partition, v_weights updated by model prediction, weights updated by all history txs:")
    graph_path = f'./metis/graphs/test_vpred_double_{window}.txt'
    txs = txs[['from','to']]
    if args.double_addr:
        base_allocator = TableAccountAllocate(n_shards=n_shards, fallback=None)
        fallback_allocator = TableAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
        allocator = DoubleAccountAllocate(n_shards=n_shards, base=base_allocator, fallback=fallback_allocator)
        client = DoubleAddrClient(txs=txs, tx_rate=tx_rate)

    else:
        allocator = TableAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
        client = Client(txs=txs, tx_rate=tx_rate)
    simulator = Eth2v3Simulator(client=client, allocate=allocator, n_shards=n_shards, n_blocks=n_blocks, tx_per_block=tx_per_block, block_interval=block_interval)
    simulator.reset()
    # first simulate window-1 epochs
    for epoch in range(window-1):
        simulator.step(None)
    log.print('Initial condition:')
    log.print(simulator.info())
    block_txs = simulator.get_block_txs()
    graph = Graph(block_txs, v_weight=False) # empty graph without v_weights
    hgraph = GraphStack(block_txs.assign(block=block_txs['block']//simulator.n_blocks))
    mse_list = []
    target_weights = np.full(n_shards, fill_value=1./n_shards, dtype=float)
    # simulate on rest txs
    for epoch in tqdm(range(window-1, simulator.max_epochs-window+1)):
        X = hgraph.get_vweight_matrix(-(window-1)).toarray()
        y_pred = model.predict(X)
        y_pred = np.maximum(np.ceil(y_pred).astype(int), 0) + 1 # convert to int>=1
        graph.set_vweight(hgraph.vertex_idx, y_pred).save(graph_path, v_weight=True)
        if args.adjust_part_weights is not None:
            n_txs = simulator.get_block_n_txs(-1)
            target_weights = np.ones_like(target_weights)
            for shard_id, shard_n_txs in enumerate(n_txs):
                if shard_n_txs<simulator.tx_per_block:
                    target_weights[shard_id] *= args.adjust_part_weights
            target_weights = target_weights/np.sum(target_weights)
        parts = Partition(graph_path).partition(args.n_shards, target_weights=target_weights)
        if args.double_addr:
            base_account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
            fallback_account_table = {a[1]:s for a,s in zip(graph.vertex_idx,parts)}
            action = (base_account_table, fallback_account_table)
        else:
            action = {a:s for a,s in zip(graph.vertex_idx,parts)}
        done = simulator.step(action)
        block_txs = simulator.get_block_txs(-simulator.n_blocks)
        graph.update(block_txs)
        hgraph.update(block_txs.assign(block=np.repeat(epoch, len(block_txs.index))))
        y_true = hgraph.get_weight_matrix(-1)[:len(y_pred),:].toarray()
        mse_list.append(mean_squared_error(y_true, y_pred))
        if done: break
    log.print('Final condition:')
    log.print(simulator.info())
    log.print('MSE:', np.average(mse_list), max(mse_list), min(mse_list))

def test_prediction(train_txs, test_txs, window=5, args=None):
    log.print('Account prediction:')
    train_txs = train_txs[['from','to']]
    test_txs = test_txs[['from','to']]
    epoch_tx_count = min(args.tx_rate*args.block_interval*args.n_blocks, args.tx_per_block*args.n_shards*args.n_blocks)

    # training model
    train_epochs = len(train_txs)//epoch_tx_count
    train_n_tx = train_epochs*epoch_tx_count
    log.print('Training prediction model:', len(train_txs), train_n_tx)
    train_hgraph = GraphStack(pd.DataFrame({'block':np.repeat(np.arange(train_epochs,dtype=int),epoch_tx_count),'from':train_txs['from'].values[:train_n_tx], 'to':train_txs['to'].values[:train_n_tx]}), debug=True)
    train_vweight_matrix = train_hgraph.get_vweight_matrix().toarray()
    log.print('Train model with data:', train_vweight_matrix.shape)
    baseline_model = AverageModel()
    model = MLPModel(base_model=baseline_model, window=window, min_value=args.min_value, normalize=args.normalize, shuffle=True, val_on_time=True, early_stop=True, tol=1e-6, tol_iters=20)
    train_score = model.fit(train_vweight_matrix)
    log.print('Train score:', train_score)
    log.print('Train loss curves:', model.loss_curves)
    log.print('Train score curves:', model.score_curves)

    log.print('Simulate on train txs:', len(train_txs))
    simulate(train_txs, model=model, window=window, args=args)

    log.print('Simulate on test txs:', len(test_txs))
    simulate(test_txs, model=model, window=window, args=args)

    log.print('Baseline model test:')
    
    log.print('Simulate on train txs:', len(train_txs))
    simulate(train_txs, model=baseline_model, window=window, args=args)

    log.print('Simulate on test txs:', len(test_txs))
    simulate(test_txs, model=baseline_model, window=window, args=args)

if __name__=='__main__':
    from exp.args import get_default_parser
    parser = get_default_parser('test prediction')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--train_start_time', type=str, default='2021-07-26 00:00:00')
    parser.add_argument('--train_end_time', type=str, default='2021-07-31 23:59:59')
    parser.add_argument('--min_value', type=float, default=0.)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--adjust_part_weights', type=float, default=None)
    parser.add_argument('--double_addr', action='store_true', default=False)
    args = parser.parse_args()
    log.print(args)

    loader = get_default_dataloader()
    _, train_txs = loader.load_data(start_time=args.train_start_time, end_time=args.train_end_time)
    _, test_txs = loader.load_data(start_time=args.start_time, end_time=args.end_time)
    drop_contract_creation_tx(train_txs)
    drop_contract_creation_tx(test_txs)

    if args.n_shards==0: args.n_shards = 1 << args.k

    test_prediction(train_txs=train_txs, test_txs=test_txs, window=args.window, args=args)
