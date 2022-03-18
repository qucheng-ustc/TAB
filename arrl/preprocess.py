import numpy as np

def drop_contract_creation_tx(txs):
    txs['from'].replace('', np.nan, inplace=True)
    txs['to'].replace('', np.nan, inplace=True)
    txs.dropna(inplace=True)

