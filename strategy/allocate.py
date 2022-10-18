
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
        # action: a list [s0, s1, s2, ... ,sm], si indicates shard id of group i
        assert(len(action)==len(self.group_table))
        self.group_table = action

    def allocate(self, addr):
        return self.group_table[addr >> self.shift]

