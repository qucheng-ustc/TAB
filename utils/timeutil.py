import time
import datetime

def strftime(timestamp=-1, fmt='%Y-%m-%d %H:%M:%S'):
    if timestamp>=0:
        return datetime.datetime.fromtimestamp(timestamp).strftime(fmt)

def strptime(timestr, fmt=['%Y-%m-%d %H:%M:%S','%Y-%m-%d']):
    if isinstance(fmt, str):
        fmt = [fmt]
    dt = None
    for tfmt in fmt:
        try:
            dt = datetime.datetime.strptime(timestr, tfmt)
        except:
            pass
    return int(dt.timestamp())

start_time = {}

def start(timer=0):
    start_time[timer] = time.time()
    return start_time[timer]

def next(timer=0):
    next_time = time.time()
    dtime = next_time - start_time[timer]
    start_time[timer] = next_time
    return dtime

def elapsed(timer=0):
    return time.time()-start_time[timer]

