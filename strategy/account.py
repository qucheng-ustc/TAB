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

# allocate an account that contains a combined addr (addr[0], addr[1])
# this strategy will try to allocate the combined addr (addr[0], addr[1]) with base strategy first
# if failed, it will allocate addr[0] with its fallback strategy instead
class DoubleAccountAllocate(AccountAllocate):
    def __init__(self, n_shards, base, fallback):
        # base: allocate strategy to allocate the combined addr (addr[0], addr[1])
        # fallback: allocate strategy to allocate the single addr addr[0]
        super().__init__(n_shards)
        self.base = base
        self.fallback = fallback
    
    def reset(self):
        self.base.reset()
        self.fallback.reset()
    
    # apply action on base strategy
    def apply(self, action):
        # action is a tuple for base, fallback strategies
        self.base.apply(action=action[0])
        self.fallback.apply(action=action[1])

    def allocate(self, addr):
        ret = self.base.allocate(addr)
        if ret!=-1:
            return ret
        return self.fallback.allocate(addr[0])
