import numpy as np
import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader

import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse
from exp.log import get_logger

from graph.stack import GraphStack

log = get_logger(file_name="./logs/test_stack_weights.log")

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

def test_stack_update(txs, step_size, debug=False):
    print('Test stack update:')
    hgraph = GraphStack()
    ntx = len(txs)
    for step in range(0, ntx, step_size):
        print('Step:', step, "/", ntx)
        stxs = txs.next(min(step_size,len(txs)-step))
        blockNumber, tx_from, tx_to = tuple(zip(*stxs))
        print('Block number:', min(blockNumber), max(blockNumber), max(blockNumber)-min(blockNumber))
        block_txs = pd.DataFrame({'block':np.repeat(step, len(stxs)), 'from':tx_from, 'to':tx_to})
        hgraph.update(block_txs, debug=debug)
    v_sets_list = []
    v_sets = [set()]+[set(v) for v in hgraph.vertexes]
    v_sets_list.append(v_sets)
    for l in range(1,10):
        vlp_sets = v_sets_list[l-1]
        vl_sets = [set()]+[vlp_sets[i-1]|v_sets[i] for i in range(1,len(v_sets))]
        v_sets_list.append(vl_sets)
    pstr = f"{len(hgraph.vertexes)}: Vertex:{[len(v) for v in hgraph.vertexes]} NewV:{[len(v) for v in hgraph.new_vertexes]}"
    for l in range(0,10):
        pstr += f" TotalV-{l+1}:{[len(v_sets_list[l][i]) for i in range(1,len(v_sets))]}" + f" NewV-{l+1}:{[len(v_sets[i]-v_sets_list[l][i-1]) for i in range(1,len(v_sets))]}"
    e_sets_list = []
    e_sets = [set()]+[set(e) for e in hgraph.edges]
    e_sets_list.append(e_sets)
    for l in range(1,10):
        elp_sets = e_sets_list[l-1]
        el_sets = [set()]+[elp_sets[i-1]|e_sets[i] for i in range(1,len(e_sets))]
        e_sets_list.append(el_sets)
    pstr += f"{len(hgraph.edges)}: Edge:{[len(e) for e in hgraph.edges]} NewE:{[len(e) for e in hgraph.new_edges]}"
    for l in range(0,10):
        pstr += f" TotalE-{l+1}:{[len(e_sets_list[l][i]) for i in range(1,len(e_sets))]}" + f" NewE-{l+1}:{[len(e_sets[i]-e_sets_list[l][i-1]) for i in range(1,len(e_sets))]}"
    log.print(pstr)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test stack')
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--step_size', type=int, default=20000)
    parser.add_argument('--start_time', type=str, default='2021-08-01 00:00:00')
    parser.add_argument('--end_time', type=str, default=None)
    parser.add_argument('--type', type=str, default='pred', choices=['pred', 'update'])
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    loader = get_default_dataloader()
    txs = loader.load_data(start_time=args.start_time, end_time=args.end_time, columns=['block_number','from','to'])

    if args.type == 'update':
        test_stack_update(txs=txs, step_size=args.step_size, debug=args.debug)
    else:
        test_stack_pred(txs=txs, window=args.window, step_size=args.step_size)
