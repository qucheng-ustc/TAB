from arrl.dataset import Dataset
from env.gym import Eth2v1

if __name__=='__main__':
    dataset = Dataset(start_time='2021-08-01 00:00:00')

    env = Eth2v1(config=dict(txs=dataset.txs, k=3, g=7, tx_rate=100))

    obs = env.reset()
    done = 0
    while not done:
        obs, reward, done, _ = env.step(None)
    print(env.info())

