import json
import time
import datetime
import random
import os

def current_timestr():
    now = datetime.datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    return timestr

def auto_name(params):
    strs = [current_timestr()]
    for param, param_value in params.items():
        paramstr = f"{param}={param_value}"
        strs.append(paramstr)
    randomstr = f"{random.randint(0,99):02}"
    strs.append(randomstr)
    return "_".join(strs)

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
        filename = self.name+".json"
        filepath = os.path.join(self.path, filename)
        print("Save record to:", filepath)
        with open(filepath, "w") as f:
            json.dump(self.record, f, indent=1)
