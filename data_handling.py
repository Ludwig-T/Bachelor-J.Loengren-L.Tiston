import cdflib
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = "C:/Users/ludwi/Desktop/Data/High Freq/06/" \
    "solo_L2_rpw-tds-surv-tswf-e-cdag_20200612_V15.cdf"
c = cdflib.cdfread.CDF(FILE_PATH)

FLAGs = [1 if flag[1] == 2 else 0 for flag in c['DOWNLINK_INFO']] #Flagged dust detection
FLAG_INDs = [i for i in range(len(FLAGs)) if FLAGs[i] == 1]       #Indicies for flagged dust

WAVEFORM = c['WAVEFORM_DATA']
SAMPLING_RATE = c['SAMPLING_RATE']

#WAVEFORM_UNITS = c.varattsget('WAVEFORM_DATA')['UNITS'] #Get attribute of variable

def tripple_plot(X, Y):
    #Plots 3 graphs on common x-axis
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    
    # Plot data on each subplot
    axes[0].plot(X, Y[0])
    axes[1].plot(X, Y[1])
    axes[2].plot(X, Y[2])
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time (ms)")
    plt.ylabel("Electric field (V/m)")
    plt.show()

for EPOCH in FLAG_INDs:
    Y = WAVEFORM[EPOCH,:,:]
    Y = [Y[0,:], Y[1,:], Y[2,:]]
    dt = (SAMPLING_RATE[EPOCH])**(-1)*1000 #from Hz to ms
    X = np.arange(0, Y[0].size*dt, dt)

    tripple_plot(X, Y)
