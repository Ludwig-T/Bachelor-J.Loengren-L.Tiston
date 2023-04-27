import cdflib
import matplotlib.pyplot as plt
import numpy as np

def tripple_plot(X, Y, to_show=True, title="Electric field (V/m)"):
    '''Plots 3 graphs on common x-axis'''
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    
    # Plot data on each subplot
    if isinstance(Y, np.ndarray):
        axes[0].plot(X, Y[0, :, 0])
        axes[1].plot(X, Y[0, :, 1])
        axes[2].plot(X, Y[0, :, 2])
    else:
        axes[0].plot(X, Y[0])
        axes[1].plot(X, Y[1])
        axes[2].plot(X, Y[2])
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (ms)")
    plt.title(title)
    if to_show:
        plt.show()
    else:
        return fig, axes

def get_data(filepath, to_extract="WAVEFORM_DATA", want_DOWNLINK_INFO=True):
    '''Input: filepath of cdf-file with data, what is to be extracted from file,
     and if it should return epoch indicies, onboard flags for dust, or "Both".
     Extracts desired data from file (e.g. waveform data), sampling rate,
     and (if wanted) downlink info from cdf file'''
     
    #Read data
    c = cdflib.cdfread.CDF(filepath)
    WAVEFORM = c[to_extract]
    SAMPLING_RATE = c['SAMPLING_RATE']
    
    if want_DOWNLINK_INFO == True:
        FLAGs = [1 if flag[1] == 2 else 0 for flag in c['DOWNLINK_INFO']]   #Flags for dust by onboard comp.
        FLAGGED_EPOCHS = [i for i in range(len(FLAGs)) if FLAGs[i] == 1]    #Indicies for flagged dust
        return WAVEFORM, SAMPLING_RATE, FLAGGED_EPOCHS
    
    elif want_DOWNLINK_INFO == 'Both':
        FLAGs = [1 if flag[1] == 2 else 0 for flag in c['DOWNLINK_INFO']]   #Flags for dust by onboard comp.
        EPOCHS = [i for i in range(len(SAMPLING_RATE))]
        return WAVEFORM, SAMPLING_RATE, EPOCHS, FLAGs 
           
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

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth