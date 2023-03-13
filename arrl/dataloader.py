import pandas as pd
import datetime

class DataLoader:
    def __init__(self, host="127.0.0.1", port=3306, user='root', db='arrl', charset='utf8'):
        import pymysql
        self.db = pymysql.connect(host=host, port=port, user=user, db=db, charset=charset)
        self.cursor = self.db.cursor()
        self.cursor.execute("select version()")
        data = self.cursor.fetchone()
        print("Database Version: %s"%data)

    def load_data(self, start_time, end_time=None):
        # start_time : timestamp or time str with format "%Y-%m-%d %H:%M:%S"
        if isinstance(start_time, int):
            pass
        elif isinstance(start_time, str):
            start_time = int(datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp())
        else:
            raise TypeError(f"{start_time} is not an int or str")
        cursor = self.cursor
        if end_time is None:
            sql = "select * from block where timestamp>=%d"%start_time
        else:
            if isinstance(end_time, int):
                pass
            elif isinstance(end_time, str):
                end_time = int(datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').timestamp())
            else:
                raise TypeError(f"{end_time} is not an int or str")
            sql = "select * from block where timestamp>=%d and timestamp<=%d"%(start_time, end_time)
        cursor.execute(sql)
        print(sql)
        data = cursor.fetchall()
        df_block = pd.DataFrame(data, columns=['blockNumber','gasLimit','gasUsed','transactionNumber','timestamp'])
        print(len(df_block))
        if end_time is None:
            sql = "select * from tx where block_number>=%d"%df_block.iloc[0,0]
        else:
            sql = "select * from tx where block_number>=%d and block_number<=%d"%(df_block.iloc[0,0],df_block.iloc[-1,0])
        print(sql)
        cursor.execute(sql)
        data = cursor.fetchall()
        df_tx = pd.DataFrame(data, columns=['blockNumber','transactionIndex','from','to','gas'])
        print(len(df_tx))
        return df_block, df_tx

class XBlockLoader:

    files = [
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

