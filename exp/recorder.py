import json
import time
import datetime
import random
import os
import re

def current_timestr():
    now = datetime.datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    return timestr

def auto_name(params):
    strs = [current_timestr()]
    for param, param_value in params.items():
        paramstr = f"{param}={param_value}"
        strs.append(paramstr)
    return "_".join(strs)

def sanitize_filename(filename):
    # 移除非法字符
    filename = re.sub(r'[\/:*?"<>|]', '', filename)
    
    # 缩短文件名
    max_length = 255  # 假设文件名最大长度为255个字符
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # 处理重复文件名
    count = 1
    new_filename = filename
    while os.path.exists(new_filename):
        base_name, extension = os.path.splitext(filename)
        new_filename = f"{base_name}_{count}{extension}"
        count += 1
    
    return new_filename


class Recorder:
    def __init__(self, path, params, name=None):
        if name is None:
            name = auto_name(params)
        self.path = path
        self.name = name
        self.params = params

        self.record = dict(
            name=self.name,
            time=current_timestr(),
            params=self.params,
            values=dict()
        )
    
    def add(self, key, value):
        self.record["values"][key]=value
    
    def save(self):
        filename = sanitize_filename(self.name+".json")
        filepath = os.path.join(self.path, filename)
        print("Save record to:", filepath)
        with open(filepath, "w") as f:
            json.dump(self.record, f, indent=1)
