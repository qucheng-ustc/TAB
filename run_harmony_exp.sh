#!/bin/bash

K=$1
TX_RATE=$2

N_EPOCHS=10

START_TIME="2021-02-01 00:00:00"
END_TIME="2021-04-01 00:00:00"

python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --double_addr --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=none --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=pending --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}

START_TIME="2021-04-01 00:00:00"
END_TIME="2021-06-01 00:00:00"

python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --double_addr --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=none --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=pending --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}

START_TIME="2021-06-01 00:00:00"
END_TIME="2021-08-01 00:00:00"

python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --double_addr --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=none --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
python test_harmony.py --method=pending --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS}
