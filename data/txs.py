import numpy as np
import pandas as pd
import requests
import json
import datetime

import jrpc

def query_data(firstNumber = 0xc17de9, lastNumber = 0xc242b4):

    blocks = []
    txs = []

    for number in range(firstNumber, lastNumber+1):
        response = jrpc.eth_jrpc("eth_getBlockByNumber",[f'0x{number:x}',True], id=number)
        assert(response.status_code == 200)
        response = json.loads(response.content)
        assert(response['id'] == number)
        block = response['result']
        gasLimit = int(block['gasLimit'], base=16)
        gasUsed = int(block['gasUsed'], base=16)
        timestamp = int(block['timestamp'], base=16)
        blocktime = datetime.datetime.utcfromtimestamp(timestamp)
        transactions = block['transactions']
        transactionNumber = len(transactions)
        print(f'0x{number:x}: txs:{transactionNumber} gas:{gasUsed}/{gasLimit} {blocktime.strftime("%Y-%m-%d %H:%M:%S")}')
        blocks.append({'blockNumber':number, 'gasLimit':gasLimit, 'gasUsed':gasUsed, 'transactionNumber':transactionNumber, 'timestamp':timestamp})
        for tx in transactions:
            transactionIndex = int(tx['transactionIndex'], base=16)
            tx_from = tx['from']
            tx_to = tx['to']
            tx_gas = int(tx['gas'], base=16)
            txs.append({'blockNumber':number, 'transactionIndex':transactionIndex, 'from':tx_from, 'to':tx_to, 'gas':tx_gas})

    df_block = pd.DataFrame(blocks, columns=['blockNumber', 'gasLimit', 'gasUsed', 'transactionNumber', 'timestamp'])
    df_tx = pd.DataFrame(txs, columns=['blockNumber', 'transactionIndex', 'from', 'to', 'gas'])

    df_block.to_csv(f'blocks_{firstNumber:x}_{lastNumber:x}.csv')
    df_tx.to_csv(f'txs_{firstNumber:x}_{lastNumber:x}.csv')


if __name__=='__main__':
    query_data(0xc17de9, 0xc242b4)
    #query_data(0xbf6cc3, 0xc17de8)

