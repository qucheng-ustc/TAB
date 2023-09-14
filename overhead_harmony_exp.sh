#!/bin/bash

run_shard_exp() {
python test_harmony.py --method=shard --k=${K} --tx_rate=${TX_RATE} --tx_per_block=2000 --block_interval=10 --n_blocks=100 --double_addr --pmatch --compress ${COMPRESS} --start_time="${START_TIME}" --end_time="${END_TIME}" --n_epochs=${N_EPOCHS} ${OVERHEAD}
}

export K=$1
export TX_RATE=$2
export N_EPOCHS=10
export OVERHEAD=--overhead

for COMPRESS_V in {11..20};do
  # for COMPRESS_S in `seq $COMPRESS_V 10`;do
    export COMPRESS="1 $COMPRESS_V $COMPRESS_V"

    echo "COMPRESS: $COMPRESS";

    export START_TIME="2021-02-01 00:00:00"
    export END_TIME="2021-04-01 00:00:00"
    run_shard_exp;

    export START_TIME="2021-04-01 00:00:00"
    export END_TIME="2021-06-01 00:00:00"
    run_shard_exp;

    export START_TIME="2021-06-01 00:00:00"
    export END_TIME="2021-08-01 00:00:00"
    run_shard_exp;
  # done
done
