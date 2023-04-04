import cdflib
import matplotlib.pyplot as plt
import numpy as np
import os

def tripple_plot(X, Y, to_show=True):
    '''Plots 3 graphs on common x-axis'''
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    
    # Plot data on each subplot
    axes[0].plot(X, Y[0])
    axes[1].plot(X, Y[1])
    axes[2].plot(X, Y[2])
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (ms)")
    plt.title("Electric field (V/m)")
    if to_show:
        plt.show()
    else:
        return fig, axes

def get_data(filepath, want_DOWNLINK_INFO=True):
    '''Extracts waveform data, sampling rate and (if wanted) downlink info from cdf file'''
    c = cdflib.cdfread.CDF(filepath)
    WAVEFORM = c['WAVEFORM_DATA']
    SAMPLING_RATE = c['SAMPLING_RATE']
    if want_DOWNLINK_INFO:
        FLAGs = [1 if flag[1] == 2 else 0 for flag in c['DOWNLINK_INFO']] #Flags for dust by onboard comp.
        FLAGGED_EPOCHS = [i for i in range(len(FLAGs)) if FLAGs[i] == 1]       #Indicies for flagged dust
        return WAVEFORM, SAMPLING_RATE, FLAGGED_EPOCHS
    else:
        EPOCHS = [i for i in range(len(SAMPLING_RATE))]
        return WAVEFORM, SAMPLING_RATE, EPOCHS
    
def gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH):
    '''For given epoch, generates waveform timeseries for each antenna based on sampling rate'''
    Y = WAVEFORM[EPOCH,:,:]
    Y = [Y[0,:], Y[1,:], Y[2,:]]
    data_size = Y[0].size
    timestep = (SAMPLING_RATE[EPOCH])**(-1)*1000 #from Hz to ms
    start_time = 0
    end_time = data_size*timestep
    times = np.arange(start_time, end_time ,timestep)
    return times, Y
#Read Data
"""
FILE_PATH = "C:/Data/High Freq/"

for root, dirs, files in os.walk(FILE_PATH):
    for file in files:
        if 'tswf' in file:
            filepath = os.path.join(root, file)
            c = cdflib.cdfread.CDF(filepath)

            FLAGs = [1 if flag[1] == 2 else 0 for flag in c['DOWNLINK_INFO']] #Flagged dust detection
            FLAG_INDs = [i for i in range(len(FLAGs)) if FLAGs[i] == 1]       #Indicies for flagged dust
            WAVEFORM = c['WAVEFORM_DATA']
            SAMPLING_RATE = c['SAMPLING_RATE']

            for EPOCH in FLAG_INDs:
                X, Y = get_data(WAVEFORM, SAMPLING_RATE, EPOCH)
                tripple_plot(X, Y)
"""
#WAVEFORM_UNITS = c.varattsget('WAVEFORM_DATA')['UNITS'] #Get attribute of variable
