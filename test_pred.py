import numpy as np
import pandas as pd
from tqdm import tqdm
from arrl.dataloader import get_default_dataloader
from arrl.preprocess import drop_contract_creation_tx

from graph.stack import GraphStack

if __name__=='__main__':
    block_txs = pd.DataFrame({'block':np.repeat(np.arange(10), 10), 'from':np.random.randint(10, size=100), 'to':np.random.randint(10, size=100)})
    hgraph = GraphStack(block_txs, debug=True)
    

