import numpy as np
import pandas as pd
import arrl.dataloader
import arrl.preprocess

class Dataset:
    def __init__(self, loader=arrl.dataloader.DataLoader(), start_time=0, end_time=None, addr_len=16):
        self.loader = loader
        self.blocks, self.txs = loader.load_data(start_time=start_time, end_time=end_time)
        arrl.preprocess.drop_contract_creation_tx(self.txs)
        arrl.preprocess.convert_address_to_int(self.txs, addr_len=16)

    def __length__(self):
        return len(self.txs)

    def __getitem__(self, index):
        return self.txs[index]

