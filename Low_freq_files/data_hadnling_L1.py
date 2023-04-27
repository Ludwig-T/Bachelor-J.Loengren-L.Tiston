import cdflib
import numpy as np
import matplotlib as plt

def get_data(filepath, to_extract="V"):
    '''Input: filepath of cdf-file with data, what is to be extracted from file
    '''
    #Read data
    c = cdflib.cdfread.CDF(filepath)
    V = c[to_extract]
    SAMPLING_RATE = c['SAMPLING_RATE']
    
    return V, SAMPLING_RATE


def get_timeseries(SAMPLING_RATE):
    '''For given epoch, generates waveform timeseries for each antenna based on sampling rate'''
    dt = [1/s for s in SAMPLING_RATE]

    times = [sum(dt[:i]) for i in range(len(dt)+1)]
    return times[1:]