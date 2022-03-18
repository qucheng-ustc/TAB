
class AllocateStrategy:
    def __init__(self, k):
        self.k = k

    def apply(self, action):
        pass
        
    def allocate(self, account):
        return 0

class RandomAllocateStrategy(AllocateStrategy):
    def __init__(self, k):
        super().__init__(k)
        self.n_chars = (self.k+3)//4
        self.shift = self.n_chars*4 - self.k

    def allocate(self, account):
        account_addr = int(account[:self.n_chars], base=16)
        return account_addr >> self.shift

class GroupAllocateStrategy(AllocateStrategy):
    def __init__(self, k, g):
        super().__init__(k)
        self.g = g
        assert(g>=k)
        self.n_chars = (self.g+3)//4
        self.shift = self.n_chars*4 - self.g

    def apply(self, table):
        self.group_table = table

    def allocate(self, account):
        account_addr = int(account[:self.n_chars], base=16)
        return self.group_table[account_addr >> self.shift]

