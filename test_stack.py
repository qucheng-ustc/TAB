import numpy as np
import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx

import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse

from graph.stack import GraphStack

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test stack')
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--step_size', type=int, default=20000)
    parser.add_argument('--start_time', type=str, default='2021-08-01 00:00:00')
    args = parser.parse_args()

    window = args.window
    step_size = args.step_size

    print('Real data:')
    loader = get_default_dataloader()
    blocks, txs = loader.load_data(start_time=args.start_time)
    drop_contract_creation_tx(txs)
    for step in range(0, len(txs)-window*step_size, (window-1)*step_size):
        print('Step:', step)
        stxs = txs.iloc[step:step+window*step_size]
        print('Block number:', min(stxs.blockNumber), max(stxs.blockNumber), max(stxs.blockNumber)-min(stxs.blockNumber))
        block_txs = pd.DataFrame({'block':np.repeat(np.arange(window), step_size), 'from':stxs['from'], 'to':stxs['to']})
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