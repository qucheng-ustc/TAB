class Tester:
    def __init__(self, txs):
        self.txs = txs
        self.tx_rate = 1000
        self.tx_per_block = 200
        self.block_interval = 15
        
    def config(self, tx_rate=1000, tx_per_block=200, block_interval=15):
        self.tx_rate = tx_rate
        self.tx_per_block = tx_per_block
        self.block_interval = block_interval
    
    def test(self, allocate_strategy, n_samples=None):
        if n_samples == None:
            txs = self.txs
        else:
            txs = self.txs.sample(n_samples)
        n_cross_tx = 0
        n_inner_tx = 0
        k = allocate_strategy.k
        n_shards = 1<<k
        n_shard_cross_tx = [0 for i in range(n_shards)]
        n_shard_inner_tx = [0 for i in range(n_shards)]
        shard_tx_pool = [[] for i in range(n_shards)]
        shard_forward_tx_pool = [[] for i in range(n_shards)]
        shard_blocks = [[] for i in range(n_shards)]
        shard_queue_length = [[] for i in range(n_shards)]
        # simulate running
        tx_count = 0
        block_count = 0
        simulate_time = 0
        for i, tx in txs.iterrows():
            s_from = allocate_strategy.allocate(tx['from_addr'])
            s_to = allocate_strategy.allocate(tx['to_addr'])
            tx = {'id':tx_count, 's_from':s_from, 's_to':s_to, 'time':simulate_time}
            shard_tx_pool[tx['s_from']].append(tx)
            tx_count += 1
            if tx_count % self.tx_rate == 0: # every 1s
                simulate_time += 1
                for i in range(n_shards):
                    tx_pool = shard_tx_pool[i]
                    shard_queue_length[i].append(len(tx_pool))
            if tx_count % (self.tx_rate * self.block_interval) == 0: # every block interval
                # generate block
                for i in range(n_shards):
                    tx_pool = shard_tx_pool[i]
                    n_tx = 0
                    n_from_tx = 0
                    n_to_tx = 0
                    for n in range(self.tx_per_block):
                        if len(tx_pool)==0:
                            break
                        tx = tx_pool.pop(0)
                        if tx['s_from'] == i:
                            n_from_tx += 1
                            if tx['s_from'] != tx['s_to']:
                                n_shard_cross_tx[i]+=1
                                n_cross_tx+=1
                                shard_forward_tx_pool[tx['s_to']].append(tx)
                            else:
                                n_shard_inner_tx[i]+=1
                                n_inner_tx+=1
                        else:
                            n_to_tx += 1
                        n_tx += 1
                    block = {'id':block_count, 'n_tx':n_tx, 'n_from_tx': n_from_tx, 'n_to_tx': n_to_tx}
                    block_count += 1
                    shard_blocks[i].append(block)
                # forward txs
                for i in range(n_shards):
                    forward_tx_pool = shard_forward_tx_pool[i]
                    while len(forward_tx_pool)>0:
                        tx = forward_tx_pool.pop(0)
                        shard_tx_pool[i].insert(0,tx)
        n_block_tx = 0
        n_from_tx = 0
        n_to_tx = 0
        n_shard_tx = [0 for i in range(n_shards)]
        for i in range(n_shards):
            for block in shard_blocks[i]:
                n_shard_tx[i] += block['n_tx']
                n_block_tx += block['n_tx']
                n_from_tx += block['n_from_tx']
                n_to_tx += block['n_to_tx']
        result = {
            'n_shards': n_shards,
            'n_tx': n_inner_tx + n_cross_tx,
            'n_inner_tx': n_inner_tx,
            'n_cross_tx': n_cross_tx,
            'n_shard_cross_tx': n_shard_cross_tx,
            'n_shard_inner_tx': n_shard_inner_tx,
            'cross_rate': n_cross_tx/(n_inner_tx+n_cross_tx),
            'simulate_time': simulate_time,
            'n_block_tx': n_block_tx,
            'n_from_tx': n_from_tx,
            'n_to_tx': n_to_tx,
            'throughput': n_block_tx/simulate_time,
            'actual_throughput': n_from_tx/simulate_time,
            'ideal_throughput': n_shards*self.tx_per_block/self.block_interval,
            'shard_queue_length': shard_queue_length
        }
        return result

