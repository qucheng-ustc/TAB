import numpy as np
import pandas as pd
import arrl.dataloader
import arrl.preprocess

class DatasetBase:
    def __length__(self):
        return len(self.txs)
    def __getitem__(self, index):
        return self.txs[index]

class Dataset(DatasetBase):
    def __init__(self, loader=None, start_time=0, end_time=None, addr_len=16):
        if loader is None:
            loader = arrl.dataloader.get_default_dataloader()
        self.loader = loader
        blocks, txs = loader.load_data(start_time=start_time, end_time=end_time)
        arrl.preprocess.drop_contract_creation_tx(txs)
        self.txs = pd.DataFrame({'from_addr':arrl.preprocess.convert_addr_to_int(txs['from'], addr_len=addr_len),
                                 'to_addr':arrl.preprocess.convert_addr_to_int(txs['to'], addr_len=addr_len)})

class RandomDataset(DatasetBase):
    def __init__(self, size, addr_len=16):
        self.txs = pd.DataFrame({'from_addr':np.random.randint(1<<addr_len, size=size),
                                 'to_addr':  np.random.randint(1<<addr_len, size=size)})
