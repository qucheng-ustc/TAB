#!/bin/bash

python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --double_addr --pmatch --compress ${COMPRESS} --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS} ${OVERHEAD}
