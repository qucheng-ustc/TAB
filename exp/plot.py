import os
import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
from exp.recorder import Recorder

class Subplots:
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(12,5))
        self.fig = fig
        self.axes = axes
    
    def subplot(self, row, col):
        return self.axes[row][col]
    
    def show(self, save_path=None):
        if self.nrows * self.ncols > 1:
            ax_idx = string.ascii_lowercase
            i = -1
            for ax_line in self.axes:
                for ax in ax_line:
                    i += 1
                    ax.set_title(f'({ax_idx[i]}) '+ax.get_title(), y=-0.2)
            self.fig.subplots_adjust(wspace=0.5)
        self.fig.tight_layout()
        if save_path is not None:
            self.fig.savefig(save_path)
        self.fig.show()

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
    'n_epochs': 10,
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
    def __init__(self, record_path, filter=None, params=None, debug=False):
        self.record_path = pathlib.Path(record_path)
        self.records = []
        for filename in self.record_path.glob("*.json"):
            if debug:
                print(filename)
            recorder = Recorder.load(filename)
            self.records.append(recorder)
        self.filter = filter
        self.params = params
        self.debug = debug

    def set_params(self, params):
        self.params = params
    
    def set_filter(self, filter):
        self.filter = filter
    
    def get_records(self, filter=None):
        filter = Filter(filter, self.filter)
        records = []
        for record in self.records:
            if filter.check(record.params):
                records.append(record)
        return records
    
    def _prepare_data(self, records, key, params=None, methods=['TAB-5,5','TAB-1,1','TAB-D','Transformers','Monoxide'], operation=None):
        if params is None:
            params = self.params
        data = {}
        for record in records:
            method = record.params['method']
            double_addr = record.params['double_addr']
            if method == 'shard':
                if double_addr:
                    compress = record.params['compress']
                    m = f'TAB-{compress[1]},{compress[2]}'
                else:
                    m = 'TAB-D'
            elif method == 'none' and not double_addr:
                m = 'Monoxide'
            elif method == 'pending' and not double_addr:
                m = 'Transformers'
            else:
                continue
            if m not in methods:
                continue
            mdata = data.setdefault(m, {})
            n_shards = record.params['n_shards']
            tx_rate = record.params['tx_rate']
            value = record.get('info')[key]
            if operation is not None:
                value = operation(value, record)
            mdata.setdefault(f"[{n_shards},{tx_rate}]",[]).append(value)
        data_dict = {}
        for i, method in enumerate(methods):
            data_list = []
            mdata = data[method]
            for param in params:
                if param in mdata:
                    datas = mdata[param]
                    if self.debug:
                        print(f'Method={method},Param={param},Key={key},Values={datas}')
                    data_list.append(np.average(datas))
            data_dict[method] = data_list
        return data_dict
    
    def _plot_bar(self, data_dict, params=None, title="", x_label="", y_label=""):
        if params is None:
            params = self.params
        plt.figure()
        n_series = len(data_dict)
        bar_width = 1./(n_series+1)
        for i, method in enumerate(data_dict):
            data_list = data_dict[method]
            plt.bar(np.arange(len(data_list))+i*bar_width, data_list, bar_width, label=method)
        plt.xticks(np.arange(len(params))+bar_width/2*len(data_dict), params)
        plt.legend()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    def plot_cross_rate(self, filter=None, params=None):
        records = self.get_records(filter)
        cross_dict = self._prepare_data(records, 'prop_cross_tx', params=params)
        self._plot_bar(cross_dict, params=params, title='Cross Rate', x_label='[K,r]', y_label='ε')
    
    def plot_utility(self, filter=None, params=None):
        records = self.get_records(filter)
        utility_dict = self._prepare_data(records, 'prop_wasted', params=params, operation=lambda x,_:1-x)
        self._plot_bar(utility_dict, params=params, title='Utility', x_label='[K,r]', y_label='U')

    def plot_actual_throughput(self, filter=None, params=None):
        records = self.get_records(filter)
        tps_dict = self._prepare_data(records, 'actual_throughput', params=params)
        self._plot_bar(tps_dict, params=params, title='Actual Throughput', x_label='[K,r]', y_label='P (TPS)')
    
    def plot_tx_delay(self, filter=None, params=None):
        records = self.get_records(filter)
        delay_dict = self._prepare_data(records, 'tx_delay', params=params)
        self._plot_bar(delay_dict, params=params, title='Transaction Delay', x_label='[K,r]', y_label='L (sec)')
    
    def plot_avg_partition_time(self, filter=None, params=None):
        records = self.get_records(filter)
        time_dict = self._prepare_data(records, 'partition_time_cost', params=params, methods=['TAB-5,5','TAB-1,1','TAB-D','Transformers'], operation=lambda x,_:sum(x)/len(x))
        self._plot_bar(time_dict, params=params, title='Avg. Partition Time', x_label='[K,r]', y_label='Partition time (sec)')
    
    def plot_avg_queue_length(self, filter=None, params=None):
        records = self.get_records(filter)
        ql_dict = self._prepare_data(records, 'pending_length_mean', params=params)
        self._plot_bar(ql_dict, params=params, title='Avg. Queue Length', x_label='[K,r]', y_label='Avg. Queue Length')
    
    def plot_queue_length_std(self, filter=None, params=None):
        records = self.get_records(filter)
        ql_dict = self._prepare_data(records, 'pending_length_std', params=params)
        self._plot_bar(ql_dict, params=params, title='Queue Length Stddev.', x_label='[K,r]', y_label='Queue Length Stddev.')

import re

class LogPloter:
    def __init__(self, log_file):
        self.log_file = log_file
    
    def _get_list_data(self, list_str):
        v_str = list_str.removeprefix('[').removesuffix(']')
        v = [int(v) for v in v_str.split(',')]
        return v

    def _read_file(self, key):
        with open(self.log_file, 'r') as f:
            for last_line in f:
                pass
            values = last_line.split('---')[-1]
        result = re.findall("([0-9A-Za-z-]+):(\[[^\]]*\])", values)
        data = {}
        if len(result)>2:
            for i, res in enumerate(result):
                data[res[0]] = self._get_list_data(res[1])[key]
        return data
    
    def _data_to_list(self, data):
        l = [0]*len(data)
        for k,v in data.items():
            name, value = k.split('-')
            value = int(value)
            l[value-1] = v
        return l
    
    def _normalize(self, data):
        data_max = max(data)
        data = [d/data_max for d in data]
        return data
    
    def _get_ax(self):
        plt.figure()
        return plt.subplot()
    
    def plot_cost_and_coverage(self, key=None, normalize=False, type='Vertex', ax=None, title=True, show=True):
        if type == 'Vertex':
            total_type = 'TotalV-'
            new_type = 'NewV-'
            title = 'Accounts Coverage and Cost'
            ylabel0 = 'Accounts Coverage'
            ylabel1 = 'Accounts Cost'
        if type == 'Edge':
            total_type = 'TotalE-'
            new_type = 'NewE-'
            title = 'Edges Coverage and Cost'
            ylabel0 = 'Edges Coverage'
            ylabel1 = 'Edges Cost'
        data = self._read_file(key)
        costs = {k:v for k,v in data.items() if k.startswith(total_type)}
        cost_list = self._data_to_list(costs)
        coverages = {k:1-v/data[type] for k,v in data.items() if k.startswith(new_type)}
        coverage_list = self._data_to_list(coverages)
        n = len(cost_list)
        if normalize:
            cost_list = self._normalize(cost_list)
        if ax is None:
            ax0 = self._get_ax()
        else:
            ax0 = ax
        print('Coverage:', coverage_list)
        ax0.bar(np.arange(n), coverage_list, 0.5, color='blue', label='Coverage')
        ax1 = ax0.twinx()
        effect_list = [coverage/cost for coverage,cost in zip(coverage_list, cost_list)]
        if normalize:
            effect_list = self._normalize(effect_list)
        print('Cost:', cost_list)
        ax1.plot(np.arange(n), cost_list, color='red', label='Cost')
        # ax1.plot(np.arange(n), effect_list, color='green', label='Efficiency')
        ax0.set_xticks(np.arange(n), np.arange(1,n+1))
        ax0.legend(loc='upper left')
        ax1.legend(loc='upper right')
        ax0.set_title(title)
        ax0.set_xlabel('φ Steps')
        ax0.set_ylabel(ylabel0)
        ax1.set_ylabel(ylabel1)
        if show:
            plt.show()

    def plot_new_accounts_ratio(self, key=None, type='Vertex'):
        if type == 'Vertex':
            new_type = 'NewV'
            title = "Ratio of New Accouts"
        if type == 'Edge':
            new_type = 'NewE'
            title = "Ratio of New Edges"
        data = self._read_file(key)
        ratios = {k:[vi[0]/vi[1] for vi in zip(v,data[type])] for k,v in data.items() if k.startswith(new_type)}
        plt.figure()
        for k, v in ratios.items():
            plt.plot(range(len(v)), v, label=k)
        plt.ylim((0,1))
        plt.title(title)
        plt.legend()
        plt.show()
    
    def plot_new_accounts(self, step_size=160000, key=None, type='Vertex'):
        if type == 'Vertex':
            new_type = 'NewV'
            title = 'Number of New Accounts'
        if type == 'Edge':
            new_type = 'NewE'
            title = 'Number of New Edges'
        data = self._read_file(key)
        plt.figure()
        for k, v in data.items():
            if k == type or k.startswith(new_type):
                plt.plot(range(len(v)), v, label=k)
        plt.ylim(bottom=0)
        plt.title(title)
        plt.legend()
        plt.show()

