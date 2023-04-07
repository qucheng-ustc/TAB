def get_default_parser(description=None):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-k', '--k', type=int, default=3)
    parser.add_argument('--tx_rate', type=int, default=100)
    parser.add_argument('--n_blocks', type=int, default=10) # number of blocks per step
    parser.add_argument('--tx_per_block', type=int, default=200)
    parser.add_argument('--block_interval', type=int, default=15)
    parser.add_argument('--start_time', type=str, default='2021-08-01 00:00:00')
    parser.add_argument('--end_time', type=str, default=None)
    default_parser = parser
    return default_parser
