import numpy as np
import pandas as pd

#allocate to n_shards shards based on str address
class AccountAllocate:
    def __init__(self, n_shards):
        assert(n_shards>0)
        self.n_shards = n_shards

    def apply(self, action):
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
        self.fallback = fallback

    def apply(self, action):
        #action is a tuple (account list, partition result)
        self.account_idx, self.part = action
        if not isinstance(self.account_idx, pd.Index):
            self.account_idx = pd.Index(self.account_idx)

    def allocate(self, addr):
        idx = self.account_idx.get_indexer([addr])[0]
        if idx>=0:
            return self.part[idx]
        else:
            # failed to allocate addr, addr not in the account list
            if self.fallback:
                return self.fallback.allocate(addr)
            else:
                return -1

# allocate a double account that contains two addr, allocate addr[0] first, if failed then allocate addr[1]
class DoubleAccountAllocate(AccountAllocate):
    def __init__(self, n_shards, base, fallback=None):
        # base: allocate strategy to allocate each addr
        # fallback: return fallback.allocate if failed to allocate
        super().__init__(n_shards)
        self.base = base
        self.fallback = fallback
    
    def apply(self, action):
        self.base.apply(action=action)

    def allocate(self, addr):
        ret = self.base.allocate(addr[0])
        if ret!=-1:
            return ret
        ret = self.base.allocate(addr[1])
        if ret!=-1:
            return ret
        return self.fallback.allocate(addr)
