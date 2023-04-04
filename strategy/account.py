import numpy as np
import pandas as pd

#allocate to n_shards shards based on str address
class AccountAllocate:
    def __init__(self, n_shards):
        assert(n_shards>0)
        self.n_shards = n_shards

    def apply(self, action):
        pass

    def reset(self):
        pass

    def allocate(self, addr):
        raise NotImplementedError

class StaticAccountAllocate(AccountAllocate):
    def __init__(self, n_shards):
        super().__init__(n_shards)
        self.n_chars = len(hex(n_shards)) - 2

    def allocate(self, addr):
        addr = int(addr[:self.n_chars], base=16)
        return addr % self.n_shards

#allocate account based on the metis graph partition result
class PartitionAccountAllocate(AccountAllocate):
    def __init__(self, n_shards, fallback=None):
        # fallback: return fallback.allocate if failed to allocate
        super().__init__(n_shards)
        self.account_table = None
        self.fallback = fallback

    def apply(self, action):
        #action is a tuple (account list, partition result)
        self.account_table = {a:s for a,s in zip(*action)}

    def reset(self):
        self.account_table = None

    def allocate(self, addr):
        if self.account_table is not None:
            if addr in self.account_table:
                return self.account_table[addr]
        # failed to allocate addr, addr not in the account list
        if self.fallback:
            return self.fallback.allocate(addr)
        return -1

#allocate account based on account->shard mapping table
class TableAccountAllocate(AccountAllocate):
    def __init__(self, n_shards, fallback=None):
        super().__init__(n_shards=n_shards)
        self.account_table = {}
        self.fallback = fallback
    
    def apply(self, action):
        self.account_table.update(action)

    def reset(self):
        self.account_table.clear()
    
    def allocate(self, addr):
        if addr in self.account_table:
            return self.account_table[addr]
        if self.fallback:
            return self.fallback.allocate(addr)
        return -1

class DirectAccountAllocate(AccountAllocate):
    def allocate(self, addr):
        return addr

# allocate a double account that contains two addr, allocate addr[0] first, if failed then allocate addr[1]
class DoubleAccountAllocate(AccountAllocate):
    def __init__(self, n_shards, base, fallback=None):
        # base: allocate strategy to allocate each addr, either a single AccountAllocate or a tuple (AccountAllocate, AccountAllocate)
        # fallback: return fallback.allocate if failed to allocate
        super().__init__(n_shards)
        if isinstance(base, tuple):
            self.base = base
        else:
            self.base = (base, base)
        self.fallback = fallback
    
    # to apply different action to each allocator, use self.base[0].apply|self.base[1].apply
    def apply(self, action):
        self.base[0].apply(action=action)
        if id(self.base[0])!=id(self.base[1]):
            self.base[1].apply(action=action)

    def allocate(self, addr):
        ret = self.base.allocate(addr[0])
        if ret!=-1:
            return ret
        ret = self.base.allocate(addr[1])
        if ret!=-1:
            return ret
        return self.fallback.allocate(addr)
