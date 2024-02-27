# Repository for SwiftShard and further investigations

## Requirements

Python 3.10, MySQL 14.14  

Dependencies:  
numpy, pandas, gym, pymysql, tqdm, matplotlib, jupyter-notebook

## Data

The data is collected from Ethereum mainnet, from block 9437184 to 12965143, and converted to MySQL tables.  
The data can be downloaded from Figshare.  

You need to create a database named "arrl" and import block and tx tables from the downloaded data.  
Database settings can be changed at ./arrl/dataloader.py  

## Files

```bash
.
├── arrl                        : Dataloader and Dataset
├── data                        : Functions for retrieving test data from Ethereum (no use, use the database now)
├── env                         : Blockchain simulation environments
├── exp                         : Functions for logging, recording, ploting
├── fonts                       : Fonts
├── graph                       : Account graph
├── images                      : Experiment result images and system design drawings
├── logs                        : Logs
├── metis                       : The Metis graph partition lib
├── notes                       : Experiment notes
├── papers                      : Related or refered works
├── prediction                  : Experiments for predicting workload by ML
├── records                     : Experiment records
├── strategy                    : The account allocation strategy (dual address)
├── utils                       : Util functions
├── README.md                   : This file 
├── __init__.py
├── ablation_harmony_exp.sh     : Script for running ablation experiments
├── exp.sh                      : Example script for running a single experiment
├── loaddata.py                 : Print test data (no use, use the database now)
├── overhead_harmony_exp.sh     : Script for running parameter experiments
├── plot.ipynb                  : Jupyter notebook for ploting results
├── plot_for_paper.ipynb        : Jupyter notebook for ploting results
├── run_harmony_exp.sh          : Script for running the performance experiments
├── test_double.py              : Test code, for dual address
├── test_env.py                 : Test code, for simulation environments
├── test_graph.py               : Test code, for various account graph
├── test_harmony.py             : Main experiment script for SwiftShard using harmony simulator
├── test_munkres.py             : Test code, for the Munkres algorithm
├── test_plot.py                : Test code, for ploting functions
├── test_popular_graph.py       : Test code, for the popular graph
├── test_prediction.py          : Test code, for the transaction prediction with ML
├── test_rl.py                  : Test code, for allocating accounts with RL
├── test_stack.py               : Test code, for the graph stack
└── test_vpred.py               : Test code, for the workload prediction with ML
```

## Instructions

### Running experiments

The test_harmony.py is the main script for running the SwiftShard experiments.
An running example can be found in exp.sh.

There are scirpts for running experiments in paper:

For example, to run the performance experiemnts with tx rate 3200 and 16 shards:

```bash
./run_harmony_exp.sh 16 3200
```

The experiment records will be written by default to "./records/test_harmony", and the experiment data generated during running is saved by default to "/tmp/harmony".

### Plotting results

See the "plot.ipynb" or "plot_for_paper.ipynb", note you need to change the parameter of RecordPloter to the actual experiment record output path.
