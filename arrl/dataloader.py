import pandas as pd
from utils.timeutil import strftime, strptime
from collections import deque
import numpy as np
import math
import itertools

class TxLoader:
    def __init__(self, cursor, min_block_number, max_block_number, tx_per_block=150, columns=['blockNumber','transactionIndex','from','to','gas'], dropna=True, prefetch=1000):
        # print("TxLoader:", min_block_number, max_block_number)
        self.cursor = cursor
        self.min_block_number = min_block_number
        self.max_block_number = max_block_number
        self.dropna = dropna
        self.cache = deque()
        self.tsql = "select %s from tx where block_number>=%d and block_number<=%d"
        self.columns=columns
        self.columns_str = ','.join(['`'+c+'`' for c in self.columns])
        count_sql = self.tsql%("count(*)", min_block_number, max_block_number)
        print(count_sql, end=" : ")
        cursor.execute(count_sql)
        count = self.cursor.fetchone()[0]
        print(count)
        self.count = count
        self.tx_per_block = tx_per_block
        
        self.block_number = min_block_number
        self.idx = 0
        self.prefetch = prefetch
    
    def __len__(self):
        return self.count
    
    def next(self, n):
        # print('Get Data:', start, stop)
        if self.idx + n > self.count:
            n = self.count - self.idx
        results = []
        for _ in range(min(n, len(self.cache))):
            results.append(self.cache.popleft())
        while len(results)<n: # need fetch from db
            # estimate number of blocks
            n_blocks = math.ceil((n-len(results))/self.tx_per_block) + self.prefetch
            start_block_number = self.block_number
            stop_block_number = min(self.block_number+n_blocks, self.max_block_number)
            get_sql = self.tsql%(self.columns_str, start_block_number, stop_block_number)
            # print(get_sql, end=" : ")
            self.cursor.execute(get_sql)
            n_valid = 0
            for n_fetch in itertools.count():
                row = self.cursor.fetchone()
                if row is None:
                    break
                if self.dropna:
                    drop = False
                    for value in row:
                        if value == '':
                            drop = True
                            break
                    if drop:
                        continue
                n_valid += 1
                if len(results)<n:
                    results.append(row)
                else:
                    self.cache.append(row)
            # print(n_fetch, n_valid)
        self.idx += len(results)
        return results

class DataLoader:
    def __init__(self, host="127.0.0.1", port=3306, user='root', db='arrl', charset='utf8'):
        import pymysql
        self.db = pymysql.connect(host=host, port=port, user=user, db=db, charset=charset)
        self.cursor = self.db.cursor()
        self.cursor.execute("select version()")
        data = self.cursor.fetchone()
        print("Database Version: %s"%data)
    
    def close(self):
        self.db.close()

    def load_data(self, start_time=None, end_time=None, columns=['from','to'], dropna=True):
        # start_time : timestamp or time str with format "%Y-%m-%d %H:%M:%S"
        def parse_time(t):
            if t is None:
                return None
            if isinstance(t, int):
                pass
            elif isinstance(t, str):
                t = strptime(t)
            else:
                raise TypeError(f"{t} is not an int or str")
            return t
        start_time = parse_time(start_time)
        end_time = parse_time(end_time)
        cursor = self.cursor
        if start_time is None and end_time is None:
            sql = "select * from block"
        elif end_time is None:
            sql = "select * from block where timestamp>=%d"%start_time
        elif start_time is None:
            sql = "select * from block where timestamp<=%d"%end_time
        else:
            sql = "select * from block where timestamp>=%d and timestamp<=%d"%(start_time, end_time)
        cursor.execute(sql)
        print(sql, end=' : ')
        data = cursor.fetchall()
        df_block = pd.DataFrame(data, columns=['blockNumber','gasLimit','gasUsed','transactionNumber','timestamp'])
        print(len(df_block))
        min_block_number = df_block.iloc[0,0]
        max_block_number = df_block.iloc[-1,0]
        return TxLoader(cursor=cursor, min_block_number=min_block_number, max_block_number=max_block_number, columns=columns, dropna=dropna)

class XBlockLoader:

    tx_files = [
        "0to999999_BlockTransaction",
        "1000000to1999999_BlockTransaction",
        "2000000to2999999_BlockTransaction",
        "3000000to3999999_BlockTransaction",
        "4000000to4999999_BlockTransaction",
        "5000000to5999999_BlockTransaction",
        "6000000to6999999_BlockTransaction",
        "7000000to7999999_BlockTransaction",
        "8000000to8999999_BlockTransaction",
        "9000000to9999999_BlockTransaction",
        "10000000to10999999_BlockTransaction",
        "11000000to11999999_BlockTransaction",
        "12000000to12999999_BlockTransaction",
        "13000000to13249999_BlockTransaction",
    ]

    def __init__(self, data_path='../xblock_eth_data/'):
        import os
        self.data_path = os.path.abspath(data_path)
        print('Data path:', self.data_path)
    
    def load_data(self, start_block, end_block=None):
        import zipfile
        pass

default_dataloader = None

def get_default_dataloader():
    global default_dataloader
    if default_dataloader is None:
        default_dataloader = DataLoader()
    return default_dataloader

