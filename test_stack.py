import numpy as np
import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader

import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse

from graph.stack import GraphStack

def test_stack_pred(txs, window, step_size):
    print('Real data:')
    for step in range(0, len(txs)-window*step_size, (window-1)*step_size):
        print('Step:', step)
        stxs = txs.next(window*step_size)
        blockNumber, tx_from, tx_to = tuple(zip(*stxs))
        print('Block number:', min(blockNumber), max(blockNumber), max(blockNumber)-min(blockNumber))
        block_txs = pd.DataFrame({'block':np.repeat(np.arange(window), step_size), 'from':tx_from, 'to':tx_to})
        hgraph = GraphStack(block_txs, debug=True)
        weight_matrix = hgraph.get_weight_matrix()
        exog = pd.DataFrame(weight_matrix[:,:-2].todense())
        endog = pd.Series(weight_matrix[:,-2].toarray()[:,0])
        ols = sm.OLS(endog, exog)
        res = ols.fit()
        print(res.summary())
        test_exog = pd.DataFrame(weight_matrix[:,1:-1].todense())
        test_endog = pd.Series(weight_matrix[:,-1].toarray()[:,0])
        pred = ols.predict(res.params, exog=test_exog)
        pred_mse = mse(test_endog, pred)
        print('MSE:', pred_mse)

    print('Random data:')
    np.random.seed(20230301)
    n_vertex = hgraph.n_vertex
    block_txs = pd.DataFrame({'block':np.repeat(np.arange(5), 20000), 'from':np.random.randint(n_vertex, size=100000), 'to':np.random.randint(n_vertex, size=100000)})
    hgraph = GraphStack(block_txs, debug=True)
    print('Graph size:', hgraph.size)
    weight_matrix = hgraph.get_weight_matrix()
    exog = pd.DataFrame(weight_matrix[:,:-1].todense())
    endog = pd.Series(weight_matrix[:,-1].toarray()[:,0])
    ols = sm.OLS(endog, exog)
    res = ols.fit()
    print(res.summary())

def test_stack_update(txs, step_size):
    print('Test stack update:')
    hgraph = GraphStack()
    for step in range(0, len(txs), step_size):
        print('Step:', step)
        stxs = txs.next(min(step_size,len(txs)-step))
        blockNumber, tx_from, tx_to = tuple(zip(*stxs))
        print('Block number:', min(blockNumber), max(blockNumber), max(blockNumber)-min(blockNumber))
        block_txs = pd.DataFrame({'block':np.repeat(step, len(stxs)), 'from':tx_from, 'to':tx_to})
        hgraph.update(block_txs, debug=True)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test stack')
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--step_size', type=int, default=20000)
    parser.add_argument('--start_time', type=str, default='2021-08-01 00:00:00')
    parser.add_argument('--end_time', type=str, default=None)
    parser.add_argument('--type', type=str, default='pred', choices=['pred', 'update'])
    args = parser.parse_args()

    loader = get_default_dataloader()
    txs = loader.load_data(start_time=args.start_time, end_time=args.end_time, columns=['block_number','from','to'])

    if args.type == 'update':
        test_stack_update(txs=txs, step_size=args.step_size)
    else:
        test_stack_pred(txs=txs, window=args.window, step_size=args.step_size)
