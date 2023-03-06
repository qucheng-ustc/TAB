import pandas as pd
import pymysql
import datetime

class DataLoader:
    def __init__(self, host="127.0.0.1", port=3306, user='root', db='arrl', charset='utf8'):
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

default_dataloader = None

def get_default_dataloader():
    global default_dataloader
    if default_dataloader is None:
        default_dataloader = DataLoader()
    return default_dataloader

