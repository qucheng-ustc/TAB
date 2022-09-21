
class AllocateStrategy:
    def __init__(self, k):
        # 2^k shards
        assert(k>=0)
        self.k = k

    def apply(self, action):
        pass
        
    def allocate(self, account):
        return 0

class StaticAllocateStrategy(AllocateStrategy):
    def __init__(self, k):
        super().__init__(k)
        self.n_chars = (self.k+3)//4
        self.shift = self.n_chars*4 - self.k

    def allocate(self, account):
        account_addr = int(account[:self.n_chars], base=16)
        return account_addr >> self.shift

class GroupAllocateStrategy(AllocateStrategy):
    def __init__(self, k, g):
        # 2^k shards, 2^g groups
        super().__init__(k)
        self.g = g
        assert(g>=k)
        self.n_chars = (self.g+3)//4
        self.shift = self.n_chars*4 - self.g
        # initial group_table, map group id to shard id
        self.group_table = [i>>(g-k) for i in range(1<<g)]

    def apply(self, action):
        # action: a list [s0, s1, s2, ... ,sm], si indicates shard id of group i
        if action is not None:
            assert(len(action)==len(self.group_table))
            self.group_table = action

    def allocate(self, account):
        account_addr = int(account[:self.n_chars], base=16)
        return self.group_table[account_addr >> self.shift]

