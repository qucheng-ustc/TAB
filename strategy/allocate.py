import numpy as np

#allocate to 1<<k shards based on int address
class AllocateStrategy:
    def __init__(self, k, addr_len=16):
        # 2^k shards
        assert(k>=0)
        assert(addr_len>=k)
        self.k = k
        self.addr_len = addr_len

    def apply(self, action):
        pass

    def allocate(self, addr):
        raise NotImplementedError

class StaticAllocateStrategy(AllocateStrategy):
    def __init__(self, k, addr_len=16):
        super().__init__(k, addr_len=addr_len)
        self.shift = addr_len - k

    def allocate(self, addr):
        return addr >> self.shift

class RandomAllocateStrategy(AllocateStrategy):
    # random hash addr
    golden_ratio = 0.61803398874989484820458683436564
    def __init__(self, k, addr_len=16):
        super().__init__(k, addr_len=addr_len)
        self.n_shards = 1<<k
        self.salt = np.random.randint(1<<addr_len)
        self.multiplier = int((1<<addr_len)*self.golden_ratio)
        self.shift = addr_len - k
        self.mask = (1<<addr_len)-1

    def apply(self, action):
        if isinstance(action, int):
            self.salt = action
        else:
            self.salt = np.random.randint(1<<self.addr_len)

    def allocate(self, addr):
        return (((addr^self.salt)*self.multiplier)&self.mask) >> self.shift

class GroupAllocateStrategy(AllocateStrategy):
    def __init__(self, k, g, addr_len=16):
        # 2^k shards, 2^g groups
        super().__init__(k, addr_len=addr_len)
        self.g = g
        assert(g>=k)
        assert(addr_len>=g)
        self.shift = addr_len - g
        # initial group_table, map group id to shard id
        self.group_table = [i>>(g-k) for i in range(1<<g)]

    def apply(self, action):
        if action is None:
            return
        # action: a list [s_0, s_1, s_2, ... ,s_m], m: 1<<g, s_i: shard id of group i
        assert(len(action)==len(self.group_table))
        self.group_table = action

    def group(self, addr):
        return addr >> self.shift

    def allocate(self, addr):
        return self.group_table[self.group(addr)]

