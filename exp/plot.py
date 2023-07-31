import os
import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from exp.recorder import Recorder

default_values = {
    "n_shards": 2,
    "tx_rate": 400,
    "n_blocks": 100,
    "tx_per_block": 2000,
    "block_interval": 10,
    "start_time": "2021-07-01 00:00:00",
    "end_time": "2021-07-15 00:00:00",
    "method": "shard",
    "double_addr": True,
    "compress": None,
    "pmatch": False,
    "overhead": False,
    "save_path": "/tmp/harmony"
}

class Filter:
    def __init__(self, *values):
        self.value_dict = {}
        for value_dict in values:
            self.update(value_dict)
    
    def items(self):
        return self.value_dict.items()

    def update(self, other):
        if other is not None:
            for k, v in other.items():
                self.value_dict[k] = v
    
    def sub(self, keys):
        for k in keys:
            if k in self.value_dict:
                del self.value_dict[k]
    
    def check(self, params):
        for k, v in params.items():
            if k not in self.value_dict:
                continue
            for vi in self.value_dict[k]:
                if not isinstance(vi, type(v)):
                    return False
                if isinstance(vi, list):
                    if len(vi)!=len(v):
                        return False
                    for i in range(len(vi)):
                        if vi[i]!=v[i]:
                            return False
                elif vi!=v:
                    return False
        return True

default_filter = Filter({k:[v] for k,v in default_values.items()})

class RecordPloter:
    def __init__(self, record_path, filter=None):
        self.record_path = pathlib.Path(record_path)
        self.records = []
        for filename in self.record_path.glob("*.json"):
            print(filename)
            recorder = Recorder.load(filename)
            self.records.append(recorder)
        self.filter=filter
    
    def get_records(self, filter=None):
        filter = Filter(filter, self.filter)
        records = []
        for record in self.records:
            if filter.check(record.params):
                records.append(record)
        return records

    def plot_actual_throughput(self, filter=None):
        records = self.get_records(filter)
        methods = ['TAB-D','TAB','Transformers','Monoxide']
        params = ["[2,400]","[4,800]","[8,1600]","[16,3200]","[32,6400]"]
        tps = {}
        for record in records:
            method = record.params['method']
            double_addr = record.params['double_addr']
            if method == 'shard':
                if double_addr:
                    mtps = tps.setdefault('TAB-D', {})
                else:
                    mtps = tps.setdefault('TAB', {})
            elif method == 'none' and not double_addr:
                mtps = tps.setdefault('Monoxide', {})
            elif method == 'pending' and not double_addr:
                mtps = tps.setdefault('Transformers', {})
            else:
                continue
            n_shards = record.params['n_shards']
            tx_rate = record.params['tx_rate']
            actual_tps = record.get('info')['actual_throughput']
            mtps[f"[{n_shards},{tx_rate}]"] = actual_tps
        plt.figure()
        for i, method in enumerate(methods):
            tps_list = []
            mtps = tps[method]
            for param in params:
                tps_list.append(mtps[param])
            plt.bar(np.arange(len(params))+i*0.2, tps_list, 0.2, label=method)
        plt.legend()
        plt.show()
    
    def plot_tx_delay(self, filter=None):
        records = self.get_records(filter)
        methods = ['TAB-D','TAB','Transformers','Monoxide']
        params = ["[2,400]","[4,800]","[8,1600]","[16,3200]","[32,6400]"]
        tps = {}
        for record in records:
            method = record.params['method']
            double_addr = record.params['double_addr']
            if method == 'shard':
                if double_addr:
                    mtps = tps.setdefault('TAB-D', {})
                else:
                    mtps = tps.setdefault('TAB', {})
            elif method == 'none' and not double_addr:
                mtps = tps.setdefault('Monoxide', {})
            elif method == 'pending' and not double_addr:
                mtps = tps.setdefault('Transformers', {})
            else:
                continue
            n_shards = record.params['n_shards']
            tx_rate = record.params['tx_rate']
            actual_tps = record.get('info')['tx_delay']
            mtps[f"[{n_shards},{tx_rate}]"] = actual_tps
        plt.figure()
        for i, method in enumerate(methods):
            tps_list = []
            mtps = tps[method]
            for param in params:
                tps_list.append(mtps[param])
            plt.bar(np.arange(len(params))+i*0.2, tps_list, 0.2, label=method)
        plt.legend()
        plt.show()