import numpy as np
import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx
from strategy.account import PartitionAccountAllocate, StaticAccountAllocate
from env.eth2 import Eth2v1Simulator
from prediction.account import LinearModel
from prediction.metrics import mean_squared_error
from graph.stack import Graph, GraphStack, WeightGraph
from graph.partition import Partition

def test_prediction(train_txs, test_txs, n_shards, tx_rate, window=5, n_blocks=10):
    print('Account prediction:')
    graph_path = './metis/graphs/test_prediction.txt'
    allocator = PartitionAccountAllocate(n_shards=n_shards, fallback=StaticAccountAllocate(n_shards=n_shards))
    test_txs = test_txs[['from','to','gas']]
    simulator = Eth2v1Simulator(txs=test_txs, allocate=allocator, n_shards=n_shards, tx_rate=tx_rate, n_blocks=n_blocks)
    epoch_tx_count = simulator.epoch_tx_count

    train_epochs = len(train_txs)//epoch_tx_count
    train_n_tx = train_epochs*epoch_tx_count
    print('Training prediction model:', len(train_txs), train_n_tx)
    train_hgraph = GraphStack(pd.DataFrame({'block':np.repeat(np.arange(train_epochs,dtype=int),epoch_tx_count),'from':train_txs['from'].values[:train_n_tx], 'to':train_txs['to'].values[:train_n_tx]}), debug=True)
    train_weight_matrix = train_hgraph.get_weight_matrix().toarray()
    print('Train model:', train_weight_matrix.shape)
    model = LinearModel()
    train_score = model.fit(train_weight_matrix, window=window)
    print('Train score:', train_score, 'Coef:', model.coef_, 'Intercept:', model.intercept_)

    print('Simulate on train txs:', len(train_txs))
    train_simulator = Eth2v1Simulator(txs=train_txs, allocate=allocator, n_shards=n_shards, tx_rate=tx_rate, n_blocks=n_blocks)
    train_simulator.reset(ptx=(window-1)*epoch_tx_count)
    mse_list = []
    real_mse_list = []
    for epoch in tqdm(range(train_simulator.max_epochs-window+1)):
        X = train_weight_matrix[:,epoch:epoch+window-1]
        y_true = train_weight_matrix[:,epoch+window-1]
        y_pred = model.predict(X)
        mse_list.append(mean_squared_error(y_true, y_pred))
        y_pred = np.ceil(y_pred).astype(int)
        real_mse_list.append(mean_squared_error(y_true, y_pred))
        graph = WeightGraph(train_hgraph.vertex_idx, train_hgraph.weight_index, y_pred).save(graph_path)
        parts = Partition(graph_path).partition(n_shards)
        account_list = graph.vertex_idx
        done = train_simulator.step((account_list, parts))
        if done: break
    print(train_simulator.info())
    print('MSE:', np.average(mse_list), max(mse_list), min(mse_list))
    print('Real MSE:', np.average(real_mse_list), max(mse_list), min(mse_list))

    print('Simulate on test txs:', len(test_txs))
    n_tx = simulator.max_epochs*epoch_tx_count
    test_hgraph = GraphStack(pd.DataFrame({'block':np.repeat(np.arange(simulator.max_epochs,dtype=int),epoch_tx_count),'from':test_txs['from'].values[:n_tx],'to':test_txs['to'].values[:n_tx]}), debug=True)
    simulator.reset(ptx=(window-1)*epoch_tx_count)
    mse_list = []
    real_mse_list = []
    for epoch in tqdm(range(simulator.max_epochs-window+1)):
        Xy = test_hgraph.get_weight_matrix(start=epoch, stop=epoch+window).toarray()
        X, y_true = Xy[:,:-1], Xy[:,-1]
        y_pred = model.predict(X)
        mse_list.append(mean_squared_error(y_true, y_pred))
        y_pred = np.ceil(y_pred).astype(int)
        real_mse_list.append(mean_squared_error(y_true, y_pred))
        graph = WeightGraph(test_hgraph.vertex_idx, test_hgraph.weight_index, y_pred).save(graph_path)
        parts = Partition(graph_path).partition(n_shards)
        account_list = graph.vertex_idx
        simulator.step((account_list, parts))
    print(simulator.info())
    print('MSE:', np.average(mse_list), max(mse_list), min(mse_list))
    print('Real MSE:', np.average(real_mse_list), max(mse_list), min(mse_list))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test prediction')
    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('--tx_rate', type=int, default=100)
    parser.add_argument('--n_blocks', type=int, default=10) # number of blocks per step
    parser.add_argument('--window', type=int, default=5)
    args = parser.parse_args()
    print(args)

    loader = get_default_dataloader()
    _, train_txs = loader.load_data(start_time='2021-07-31 00:00:00', end_time='2021-07-31 23:59:59')
    _, test_txs = loader.load_data(start_time='2021-08-01 00:00:00')
    drop_contract_creation_tx(train_txs)
    drop_contract_creation_tx(test_txs)

    k = args.k
    n_shards = 1 << k
    tx_rate = args.tx_rate
    n_blocks = args.n_blocks
    window = args.window

    test_prediction(train_txs=train_txs, test_txs=test_txs, n_shards=n_shards, window=window, tx_rate=tx_rate, n_blocks=n_blocks)
