import os
import json
import pathlib
from exp.recorder import Recorder

class RecordPloter:
    def __init__(self, record_path):
        self.record_path = pathlib.Path(record_path)
        self.records = []
        for filename in self.record_path.glob("*.json"):
            print(filename)
            recorder = Recorder.load(filename)
            self.records.append(recorder)
