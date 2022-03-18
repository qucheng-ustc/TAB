import numpy as np
import pandas as pd

def load_data(first_number=0xc17de9, last_number=0xc242b4):
    df_blocks = pd.read_csv(f'data/blocks_{first_number:x}_{last_number:x}.csv', index_col=0)
    df_txs = pd.read_csv(f'data/txs_{first_number:x}_{last_number:x}.csv', index_col=0)
    return df_blocks, df_txs

if __name__=='__main__':
    df_blocks, df_txs = load_data(0xc17de9, 0xc242b4)
    print(df_blocks)
    print(df_txs)

