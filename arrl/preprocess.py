import numpy as np

def drop_contract_creation_tx(txs):
    l = len(txs)
    txs['from'].replace('', np.nan, inplace=True)
    txs['to'].replace('', np.nan, inplace=True)
    txs.dropna(inplace=True)
    print(f'dropped {l-len(txs)} contract creation tx, remaining:', len(txs))

def convert_tx_addr(txs, addr_len=16):
    txs['from_addr'] = convert_addr_to_int(txs['from'], addr_len=addr_len)
    txs['to_addr'] = convert_addr_to_int(txs['to'], addr_len=addr_len)

def convert_addr_to_int(addrs, addr_len=16):
    # address_length : reserve top n bits of the account address
    n_chars = (addr_len+3)//4
    shift = n_chars*4 - addr_len
    max_addr = (1<<addr_len) - 1
    def _convert(addr):
        result = int(addr[:n_chars], base=16)>>shift
        if result>max_addr:
            print('Address overflow:', addr, result)
        return result
    result = addrs.apply(_convert)
    print(f'convert address to int{addr_len}, reserve {n_chars} chars, shift={shift}, total:', len(result))
    return result