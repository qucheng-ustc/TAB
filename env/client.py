import numpy as np

class Client:
    def __init__(self, txs, tx_rate=1000):
        self.tx_rate = tx_rate
        self.txs = txs
        if "from_addr" in txs.columns:
            self.from_col = "from_addr"
        elif "from" in txs.columns:
            self.from_col = "from"
        else:
            raise ValueError(txs.columns)
        if "to_addr" in txs.columns:
            self.to_col = "to_addr"
        elif "to" in txs.columns:
            self.to_col = "to"
        else:
            raise ValueError(txs.columns)
        self.txs = self.txs.rename(columns={self.from_col:'from', self.to_col:'to'})[['from', 'to', 'gas']]
    
    def reset(self, ptx=0):
        self.reset_ptx = ptx
        self.ptx = ptx
        return self.ptx>=len(self.txs)
    
    def n_tx(self):
        return self.ptx - self.reset_ptx
    
    def done(self, time_interval=0):
        return self.ptx+self.tx_rate*time_interval>=len(self.txs)
    
    def next(self, time_interval, peek=False):
        txs = self.txs.iloc[self.ptx:min(self.ptx+self.tx_rate*time_interval, len(self.txs))]
        if not peek:
            self.ptx += len(txs)
        return txs
    
    def past(self, time_interval):
        txs = self.txs.iloc[self.ptx-self.tx_rate*time_interval:self.ptx]
        return txs

class PryClient(Client):
    def __init__(self, txs, tx_rate=1000, n_shards=1, account_table=None):
        super().__init__(txs=txs, tx_rate=tx_rate)
        self.n_shards = n_shards
        self.n_chars = len(hex(n_shards)) - 2
        self.format_shard_id = f"{{:0{self.n_chars}x}}"
        # client pries accounts' allocation info from this table addr->shard_id
        self.account_table = account_table
        # a map from origin account addr to [shard_id | addr] format account
        self.account_map = {}

    def combine_addr(self, shard_id, addr):
        prefix = self.format_shard_id.format(shard_id)
        return prefix + addr
    
    def new_addr(self, addr_old, addr_other):
        if addr_other in self.account_table:
            # addr_other allocated in account table
            shard_id = self.account_table[addr_other]
            return self.combine_addr(shard_id, addr_old)
        else:
            # set shard_id to be the same with addr_other
            return addr_other[:self.n_chars]+addr_old

    def new_tx(self, tx):
        addr_from = tx['from']
        addr_to = tx['to']
        new_from = self.account_map.get(addr_from, None)
        new_to = self.account_map.get(addr_to, None)
        # neither are new accounts
        if new_from and new_to:
            pass
        # both are new accounts, set a random shard_id to them
        elif not new_from and not new_to:
            shard_id = np.random.randint(self.n_shards)
            new_from = self.combine_addr(shard_id, addr_from)
            new_to = self.combine_addr(shard_id, addr_to)
            self.account_map[addr_from] = new_from
            self.account_map[addr_to] = new_to
        # only addr_from is new
        elif new_to:
            new_from = self.new_addr(addr_from, addr_to)
            self.account_map[addr_from] = new_from
        # only addr_to is new
        else:
            new_to = self.new_addr(addr_to, addr_from)
            self.account_map[addr_to] = tx['to']
        tx['from'] = new_from
        tx['to'] = new_to
        return tx

    def next(self, time_interval, peek=False):
        txs = self.txs.iloc[self.ptx:min(self.ptx+self.tx_rate*time_interval, len(self.txs))]
        if not peek:
            self.ptx += len(txs)
        return txs.apply(lambda tx:self.new_tx(tx),axis=1)
