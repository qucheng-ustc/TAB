from arrl.dataset import Dataset
from env import Eth2v1

if __name__=='__main__':
    dataset = Dataset(start_time='2021-08-01 00:00:00')

    env = Eth2v1(config=dict(txs=dataset.txs))

    _, _, done, _ = env.reset()
    while not done:
        observation, reward, done, _ = env.step(None)
    print(env.info())


