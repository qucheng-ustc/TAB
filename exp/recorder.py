import json
import datetime
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

def sanitize_filename(path, filename):
    # 移除非法字符
    filename = re.sub('''[\/:*?'"<>| \[\]]''', '', filename)
    
    # 缩短文件名
    max_length = 253  # 假设文件名最大长度为253个字符
    if len(filename) > max_length:
        base_name, extension = os.path.splitext(filename)
        base_name = base_name[:max_length-len(extension)]
        filename = base_name+extension
    
    # 处理重复文件名
    filepath = os.path.join(path, filename)
    count = 1
    while os.path.exists(filepath):
        base_name, extension = os.path.splitext(filename)
        filename = f"{base_name}_{count}{extension}"
        filepath = os.path.join(path, filename)
        count += 1
    
    return filepath

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
        filepath = sanitize_filename(self.path, self.name+".json")
        print("Save record to:", filepath)
        with open(filepath, "w") as f:
            json.dump(self.record, f, indent=1)
