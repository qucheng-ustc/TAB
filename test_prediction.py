import numpy as np
import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from strategy.account import PartitionAccountAllocate, StaticAccountAllocate, TableAccountAllocate, DoubleAccountAllocate
from env.eth2 import Eth2v1Simulator, Eth2v2Simulator
from prediction.account import LinearModel, AverageModel, LRModel, MLPModel
from prediction.metrics import mean_squared_error
from graph.stack import Graph, GraphStack, WeightGraph
from graph.partition import Partition
from env.client import Client, PryClient, DoubleAddrClient

def simulate(simulator, model, window, args):
    graph_path = './metis/graphs/test_prediction_simulate.txt'
    train_allocator = TableAccountAllocate(n_shards=args.n_shards, fallback=StaticAccountAllocate(n_shards=args.n_shards))
    train_client = Client(txs=train_txs, tx_rate=args.tx_rate)
    train_simulator = Eth2v2Simulator(client=train_client, allocate=train_allocator, n_shards=args.n_shards, n_blocks=args.n_blocks, tx_per_block=args.tx_per_block, block_interval=args.block_interval)
    train_simulator.reset()
    mse_list = []
    baseline_mse_list = []
    # first simulate window-1 epochs
    for epoch in range(window-1):
        train_simulator.step({})
    block_txs = train_simulator.get_block_txs()
    train_hgraph = GraphStack(block_txs.assign(block=block_txs['block']//train_simulator.n_blocks))
    # simulate on rest txs
    for epoch in tqdm(range(window-1, train_simulator.max_epochs-window+1)):
        X = train_hgraph.get_weight_matrix(-(window-1)).toarray()
        y_pred = model.predict(X)
        y_pred = np.ceil(y_pred).astype(int)
        graph = WeightGraph(train_hgraph.vertex_idx, train_hgraph.weight_index, y_pred).save(graph_path)
        parts = Partition(graph_path).partition(args.n_shards)
        account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
        done = train_simulator.step(account_table)
        block_txs = train_simulator.get_block_txs(-train_simulator.n_blocks)
        train_hgraph.update(block_txs.assign(block=np.repeat(epoch, len(block_txs.index))))
        y_true = train_hgraph.get_weight_matrix(-1)[:len(y_pred),:].toarray()
        mse_list.append(mean_squared_error(y_true, y_pred))
        if done: break
    print(train_simulator.info())
    print('MSE:', np.average(mse_list), max(mse_list), min(mse_list))
    print('Baseline MSE:', np.average(baseline_mse_list), max(baseline_mse_list), min(baseline_mse_list))

def test_prediction(train_txs, test_txs, window=5, args=None):
    print('Account prediction:')
    graph_path = './metis/graphs/test_prediction.txt'
    test_allocator = TableAccountAllocate(n_shards=args.n_shards, fallback=StaticAccountAllocate(n_shards=args.n_shards))
    test_txs = test_txs[['from','to','gas']]
    test_client = Client(txs=test_txs, tx_rate=args.tx_rate)
    test_simulator = Eth2v2Simulator(client=test_client, allocate=test_allocator, n_shards=args.n_shards, n_blocks=args.n_blocks, tx_per_block=args.tx_per_block, block_interval=args.block_interval)
    epoch_tx_count = test_simulator.epoch_tx_count

    # training model
    train_epochs = len(train_txs)//epoch_tx_count
    train_n_tx = train_epochs*epoch_tx_count
    print('Training prediction model:', len(train_txs), train_n_tx)
    train_hgraph = GraphStack(pd.DataFrame({'block':np.repeat(np.arange(train_epochs,dtype=int),epoch_tx_count),'from':train_txs['from'].values[:train_n_tx], 'to':train_txs['to'].values[:train_n_tx]}), debug=True)
    train_weight_matrix = train_hgraph.get_weight_matrix().toarray()
    print('Train model:', train_weight_matrix.shape)
    model = MLPModel()
    baseline_model = AverageModel()
    train_score = model.fit(train_weight_matrix, window=window)
    print('Train score:', train_score)

    print('Simulate on train txs:', len(train_txs))
    train_allocator = TableAccountAllocate(n_shards=args.n_shards, fallback=StaticAccountAllocate(n_shards=args.n_shards))
    train_client = Client(txs=train_txs, tx_rate=args.tx_rate)
    train_simulator = Eth2v2Simulator(client=train_client, allocate=train_allocator, n_shards=args.n_shards, n_blocks=args.n_blocks, tx_per_block=args.tx_per_block, block_interval=args.block_interval)
    train_simulator.reset()
    mse_list = []
    baseline_mse_list = []
    # first simulate window-1 epochs
    for epoch in range(window-1):
        train_simulator.step({})
    block_txs = train_simulator.get_block_txs()
    train_hgraph = GraphStack(block_txs.assign(block=block_txs['block']//train_simulator.n_blocks))
    # simulate on rest txs
    for epoch in tqdm(range(window-1, train_simulator.max_epochs-window+1)):
        X = train_hgraph.get_weight_matrix(-(window-1)).toarray()
        y_pred = model.predict(X)
        y_pred = np.ceil(y_pred).astype(int)
        baseline_y_pred = baseline_model.predict(X)
        baseline_y_pred = np.ceil(baseline_y_pred).astype(int)
        graph = WeightGraph(train_hgraph.vertex_idx, train_hgraph.weight_index, y_pred).save(graph_path)
        parts = Partition(graph_path).partition(args.n_shards)
        account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
        done = train_simulator.step(account_table)
        block_txs = train_simulator.get_block_txs(-train_simulator.n_blocks)
        train_hgraph.update(block_txs.assign(block=np.repeat(epoch, len(block_txs.index))))
        y_true = train_hgraph.get_weight_matrix(-1)[:len(y_pred),:].toarray()
        mse_list.append(mean_squared_error(y_true, y_pred))
        baseline_mse_list.append(mean_squared_error(y_true, baseline_y_pred))
        if done: break
    print(train_simulator.info())
    print('MSE:', np.average(mse_list), max(mse_list), min(mse_list))
    print('Baseline MSE:', np.average(baseline_mse_list), max(baseline_mse_list), min(baseline_mse_list))

    print('Simulate on test txs:', len(test_txs))
    test_hgraph = GraphStack()
    test_simulator.reset()
    for epoch in range(window-1):
        test_simulator.step({})
        test_hgraph.update(test_simulator.get_block_txs().assign(block=epoch))
    mse_list = []
    baseline_mse_list = []
    for epoch in tqdm(range(test_simulator.max_epochs-window+1)):
        X = test_hgraph.get_weight_matrix(-(window-1)).toarray()
        y_pred = model.predict(X)
        y_pred = np.ceil(y_pred).astype(int)
        baseline_y_pred = baseline_model.predict(X)
        baseline_y_pred = np.ceil(baseline_y_pred).astype(int)
        graph = WeightGraph(test_hgraph.vertex_idx, test_hgraph.weight_index, y_pred).save(graph_path)
        parts = Partition(graph_path).partition(n_shards)
        account_table = {a:s for a,s in zip(graph.vertex_idx,parts)}
        done = test_simulator.step(account_table)
        block_txs = train_simulator.get_block_txs(-train_simulator.n_blocks)
        test_hgraph.update(block_txs.assign(block=np.repeat(epoch, len(block_txs.index))))
        y_true = test_hgraph.get_weight_matrix(-1)[:len(y_pred),:].toarray()
        mse_list.append(mean_squared_error(y_true, y_pred))
        baseline_mse_list.append(mean_squared_error(y_true, baseline_y_pred))
        if done: break
    print(test_simulator.info())
    print('MSE:', np.average(mse_list), max(mse_list), min(mse_list))
    print('Baseline MSE:', np.average(baseline_mse_list), max(baseline_mse_list), min(baseline_mse_list))

if __name__=='__main__':
    from exp.args import get_default_parser
    parser = get_default_parser('test prediction')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--train_start_time', type=str, default='2021-07-26 00:00:00')
    parser.add_argument('--train_end_time', type=str, default='2021-07-31 23:59:59')
    parser.add_argument('--client', type=str, choices=['normal','pry'], default='normal')
    parser.add_argument('--simulator', type=str, choices=['eth2v1','eth2v2'], default='eth2v1')
    args = parser.parse_args()
    print(args)

    loader = get_default_dataloader()
    _, train_txs = loader.load_data(start_time=args.train_start_time, end_time=args.train_end_time)
    _, test_txs = loader.load_data(start_time=args.start_time, end_time=args.end_time)
    drop_contract_creation_tx(train_txs)
    drop_contract_creation_tx(test_txs)

    args.n_shards = 1 << args.k

    test_prediction(train_txs=train_txs, test_txs=test_txs, window=args.window, args=args)
