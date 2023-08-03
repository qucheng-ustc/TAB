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
    def __init__(self, record_path, filter=None, params=["[2,400]","[4,800]","[8,1600]","[16,3200]","[32,6400]"]):
        self.record_path = pathlib.Path(record_path)
        self.records = []
        for filename in self.record_path.glob("*.json"):
            print(filename)
            recorder = Recorder.load(filename)
            self.records.append(recorder)
        self.filter = filter
        self.params = params
    
    def get_records(self, filter=None):
        filter = Filter(filter, self.filter)
        records = []
        for record in self.records:
            if filter.check(record.params):
                records.append(record)
        return records
    
    def _prepare_data(self, records, key, params):
        methods = ['TAB-D','TAB','Transformers','Monoxide']
        data = {}
        for record in records:
            method = record.params['method']
            double_addr = record.params['double_addr']
            if method == 'shard':
                if double_addr:
                    mdata = data.setdefault('TAB-D', {})
                else:
                    mdata = data.setdefault('TAB', {})
            elif method == 'none' and not double_addr:
                mdata = data.setdefault('Monoxide', {})
            elif method == 'pending' and not double_addr:
                mdata = data.setdefault('Transformers', {})
            else:
                continue
            n_shards = record.params['n_shards']
            tx_rate = record.params['tx_rate']
            value = record.get('info')[key]
            mdata.setdefault(f"[{n_shards},{tx_rate}]",[]).append(value)
        data_dict = {}
        for i, method in enumerate(methods):
            data_list = []
            mdata = data[method]
            for param in params:
                if param in mdata:
                    datas = mdata[param]
                    print(f'Method={method},Param={param},Key={key},Values={datas}')
                    data_list.append(np.average(datas))
            data_dict[method] = data_list
        return data_dict
    
    def _plot_bar(self, data_dict, params):
        plt.figure()
        for i, method in enumerate(data_dict):
            data_list = data_dict[method]
            plt.bar(np.arange(len(data_list))+i*0.2, data_list, 0.2, label=method)
        plt.xticks(np.arange(len(params))+0.1*len(data_dict), params)
        plt.legend()
        plt.show()
    
    def plot_cross_rate(self, filter=None):
        records = self.get_records(filter)
        cross_dict = self._prepare_data(records, 'prop_cross_tx', params=self.params)
        self._plot_bar(cross_dict, params=self.params)
    
    def plot_utility(self, filter=None):
        records = self.get_records(filter)
        waste_dict = self._prepare_data(records, 'prop_wasted', params=self.params)
        utility_dict = {k:[1.-i for i in v] for k,v in waste_dict.items()}
        self._plot_bar(utility_dict, params=self.params)

    def plot_actual_throughput(self, filter=None):
        records = self.get_records(filter)
        tps_dict = self._prepare_data(records, 'actual_throughput', params=self.params)
        self._plot_bar(tps_dict, params=self.params)
    
    def plot_tx_delay(self, filter=None):
        records = self.get_records(filter)
        delay_dict = self._prepare_data(records, 'tx_delay', params=self.params)
        self._plot_bar(delay_dict, params=self.params)