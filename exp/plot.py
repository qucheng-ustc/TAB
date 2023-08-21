import os
import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
from exp.recorder import Recorder

class Subplots:
    def __init__(self, nrows=1, ncols=1, figsize=None):
        self.nrows = nrows
        self.ncols = ncols
        if figsize is None:
            figsize = (7*ncols-2, 5*nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=figsize)
        self.fig = fig
        self.axes = axes
    
    def subplot(self, row, col):
        return self.axes[row][col]
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.axes[key//self.ncols][key%self.ncols]
        if isinstance(key, tuple):
            return self.axes[key[0]][key[1]]
    
    def show(self, save_path=None, wspace=0.5, hspace=0.):
        if self.nrows * self.ncols > 1:
            ax_idx = string.ascii_lowercase
            i = -1
            for ax_line in self.axes:
                for ax in ax_line:
                    i += 1
                    ax.set_title(f'({ax_idx[i]}) '+ax.get_title(), y=-0.2)
            self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
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

def plot_func(func):
    def wrapper_func(*args, **kwargs):
        show = False
        if 'ax' not in kwargs or kwargs['ax'] is None:
            plt.figure()
            kwargs['ax'] = plt.subplot()
            show = True
        func(*args, **kwargs)
        if show:
            plt.show()
    return wrapper_func

class Ploter:
    pass

class RecordPloter(Ploter):
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
    
    @plot_func
    def _plot_bar(self, data_dict, params=None, ax=None, title="", x_label="", y_label="", **kwargs):
        if params is None:
            params = self.params
        print('Params:', params)
        n_series = len(data_dict)
        bar_width = 1./(n_series+1)
        for i, method in enumerate(data_dict):
            data_list = data_dict[method]
            print(method, ':', data_list)
            ax.bar(np.arange(len(data_list))+i*bar_width, data_list, bar_width, label=method)
        ax.set_xticks(np.arange(len(params))+bar_width/2*len(data_dict), params)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    def plot_cross_rate(self, filter=None, params=None, title='Cross Rate', **kwargs):
        records = self.get_records(filter)
        cross_dict = self._prepare_data(records, 'prop_cross_tx', params=params)
        self._plot_bar(cross_dict, params=params, title=title, x_label='[K,r]', y_label='ε', **kwargs)
    
    def plot_utility(self, filter=None, params=None, title='Utility', **kwargs):
        records = self.get_records(filter)
        utility_dict = self._prepare_data(records, 'prop_wasted', params=params, operation=lambda x,_:1-x)
        self._plot_bar(utility_dict, params=params, title=title, x_label='[K,r]', y_label='U', **kwargs)

    def plot_actual_throughput(self, filter=None, params=None, title='Actual Throughput', **kwargs):
        records = self.get_records(filter)
        tps_dict = self._prepare_data(records, 'actual_throughput', params=params)
        self._plot_bar(tps_dict, params=params, title=title, x_label='[K,r]', y_label='P (TPS)', **kwargs)
    
    def plot_tx_delay(self, filter=None, params=None, title='Transaction Delay', **kwargs):
        records = self.get_records(filter)
        delay_dict = self._prepare_data(records, 'tx_delay', params=params)
        self._plot_bar(delay_dict, params=params, title=title, x_label='[K,r]', y_label='L (sec)', **kwargs)
    
    def plot_avg_partition_time(self, filter=None, params=None, title='Avg. Partition Time', **kwargs):
        records = self.get_records(filter)
        time_dict = self._prepare_data(records, 'partition_time_cost', params=params, methods=['TAB-5,5','TAB-1,1','TAB-D','Transformers'], operation=lambda x,_:sum(x)/len(x))
        self._plot_bar(time_dict, params=params, title=title, x_label='[K,r]', y_label='Partition time (sec)', **kwargs)
    
    def plot_avg_queue_length(self, filter=None, params=None, title='Avg. Queue Length', **kwargs):
        records = self.get_records(filter)
        ql_dict = self._prepare_data(records, 'pending_length_mean', params=params)
        self._plot_bar(ql_dict, params=params, title=title, x_label='[K,r]', y_label='Avg. Queue Length', **kwargs)
    
    def plot_queue_length_std(self, filter=None, params=None, title='Queue Length Stddev.', **kwargs):
        records = self.get_records(filter)
        ql_dict = self._prepare_data(records, 'pending_length_std', params=params)
        self._plot_bar(ql_dict, params=params, title=title, x_label='[K,r]', y_label='Queue Length Stddev.', **kwargs)
    
    def plot_state_migration_size(self, filter=None, params=None, title='Total Size of Migrated State', **kwargs):
        records = self.get_records(filter)
        ss_dict = self._prepare_data(records, 'state_size_total_cost', params=params)
        self._plot_bar(ss_dict, params=params, title=title, x_label='[K,r]', y_label='State Size (Byte)')
    
    def plot_allocation_table_size(self, filter=None, params=None, title='Allocation Table Size', **kwargs):
        records = self.get_records(filter)
        at_dict = self._prepare_data(records, 'allocation_table_total_cost', params=params)
        self._plot_bar(at_dict, params=params, title=title, x_label='[K,r]', y_label='Allcation Table Size (Byte)', **kwargs)

    def plot_local_graph_size(self, filter=None, params=None, title='Local Graph Size', **kwargs):
        records = self.get_records(filter)
        lg_dict = self._prepare_data(records, 'local_graph_total_cost', params=params)
        self._plot_bar(lg_dict, params=params, title=title, x_label='[K,r]', y_label='Local Graph Size (Byte)', **kwargs)

import re

class LogPloter(Ploter):
    def __init__(self, log_file):
        self.log_file = log_file
    
    def _get_list_data(self, list_str):
        v_str = list_str.removeprefix('[').removesuffix(']')
        v = [int(v) for v in v_str.split(',')]
        return v

    def _read_file(self, key):
        if isinstance(key, tuple):
            key = slice(*key)
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
    
    def _data_to_list(self, data, prefix=None):
        if prefix is not None:
            data = {k:v for k,v in data.items() if k.startswith(prefix)}
        l = [0]*len(data)
        for k,v in data.items():
            if '-' in k:
                name, value = k.split('-')
            else:
                value = int(k)
            value = int(value)
            l[value-1] = v
        return l
    
    def _normalize(self, data):
        data_max = max(data)
        data = [d/data_max for d in data]
        return data
    
    @plot_func
    def plot_vpe_cost_and_coverage(self, key=None, ax=None):
        data = self._read_file(key)
        vcosts = self._data_to_list(data, prefix='TotalV-')
        ecosts = self._data_to_list(data, prefix='TotalE-')
        cost_list = [v+e for v,e in zip(vcosts,ecosts)]
        vcoverages = {k:1-v/data['Vertex'] for k,v in data.items() if k.startswith('NewV-')}
        vcoverage_list = self._data_to_list(vcoverages)
        ecoverages = {k:1-v/data['Edge'] for k,v in data.items() if k.startswith('NewE-')}
        ecoverage_list = self._data_to_list(ecoverages)
        n = len(cost_list)
        ax0 = ax
        bar_width = 0.3
        print('VCoverage:', vcoverage_list)
        ax0.bar(np.arange(n), vcoverage_list, bar_width, color='blue', label='Vertexes Coverage')
        print('ECoverage:', ecoverage_list)
        ax0.bar(np.arange(n)+bar_width, ecoverage_list, bar_width, color='green', label='Edges Coverage')
        ax1 = ax0.twinx()
        print('Cost:', cost_list)
        ax1.plot(np.arange(n), cost_list, color='red', label='Cost')
        ax0.set_xticks(np.arange(n)+bar_width/2, np.arange(1,n+1))
        ax0.legend(loc='upper left')
        ax1.legend(loc='upper right')
        ax0.set_title('Account Graph Cost and Coverage')
        ax0.set_xlabel('φ')
        ax0.set_ylabel('Coverage')
        ax1.set_ylabel('Cost')
    
    @plot_func
    def plot_new_ve_ratio(self, key=None, ax=None):
        data = self._read_file(key)
        vratios = [vi[0]/vi[1] for vi in zip(data['NewV'],data['Vertex'])]
        eratios = [vi[0]/vi[1] for vi in zip(data['NewE'],data['Edge'])]
        ax.plot(range(len(vratios)), vratios, label='New Vertexes', color='blue')
        ax.plot(range(len(eratios)), eratios, label='New Edges', color='green')
        ax.set_ylim((0,1))
        ax.set_title('Ratio of New Vertexes and Edges')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Ratio')
        ax.legend()
    
    @plot_func
    def plot_cost_and_coverage(self, key=None, type='Vertex', ax=None):
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
        cost_list = self._data_to_list(data, prefix=total_type)
        coverages = {k:1-v/data[type] for k,v in data.items() if k.startswith(new_type)}
        coverage_list = self._data_to_list(coverages)
        n = len(cost_list)
        ax0 = ax
        print('Coverage:', coverage_list)
        ax0.bar(np.arange(n), coverage_list, 0.5, color='blue', label='Coverage')
        ax1 = ax0.twinx()
        print('Cost:', cost_list)
        ax1.plot(np.arange(n), cost_list, color='red', label='Cost')
        ax0.set_xticks(np.arange(n), np.arange(1,n+1))
        ax0.legend(loc='upper left')
        ax1.legend(loc='upper right')
        ax0.set_title(title)
        ax0.set_xlabel('φ Steps')
        ax0.set_ylabel(ylabel0)
        ax1.set_ylabel(ylabel1)
    
    @plot_func
    def plot_new_accounts_ratio(self, key=None, type='Vertex', ax:plt.Axes=None):
        if type == 'Vertex':
            new_type = 'NewV'
            title = "Ratio of New Accouts"
        if type == 'Edge':
            new_type = 'NewE'
            title = "Ratio of New Edges"
        data = self._read_file(key)
        ratios = {k:[vi[0]/vi[1] for vi in zip(v,data[type])] for k,v in data.items() if k==new_type}
        for k, v in ratios.items():
            ax.plot(range(len(v)), v, label=k)
        ax.set_ylim((0,1))
        ax.set_title(title)
        ax.legend()
    
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

