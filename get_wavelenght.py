from data_handling import tripple_plot
import scipy.signal as scisig
import numpy as np
import matplotlib.pyplot as plt

def get_impact_time(times, DATA, reverse=False, thresh_above=0.05, thresh_below = 0.005):
    '''Input is timeseries of antennas. Output is each antennas wavelenghts start and end time.
    If reverse is True then it returns the end time of impact'''
    
    '''It checks if 5 values in a row is above thresh_above, in that case it marks the first value before those 5
    that is below thresh_below'''
    #Initialize list
    impact_time_per_antenna = []
    #Get total max for normalizing data later
    mx = np.max(DATA)

    if reverse:
        times = np.flip(times)

    for data in DATA:
        #Process deta:
        #Remove bias
        F = 21
        flattened_data = scisig.medfilt(data, kernel_size=F)
        data_median = np.median(flattened_data)
        data_nobias = data - data_median
        #Filter data
        data_filtered = scisig.medfilt(data_nobias, kernel_size=9)
        #Normalize
        data_normalized = data_filtered/mx
        
        if reverse:
            data_normalized = np.flip(data_normalized)

        # Initialize variables
        count = 0
        last_below_index = 0

        # Loop through values
        for i, v in enumerate(data_normalized):
            if np.abs(v) > thresh_above:
                count += 1
                if count == 5:
                    # Find index of closest value before the 5 consecutive values that is below the lower threshold
                    before_values = data_normalized[:i]
                    before_times = times[:i]
                    before_below = np.where(np.array(before_values) < thresh_below)[0]
                    if len(before_below) > 0:
                        last_below_index = before_below[-1]
                    break
            else:
                count = 0
        impact_time_per_antenna.append(times[last_below_index])
    return impact_time_per_antenna
                
        
def get_wavelenght(times, DATA, thresh_below=0.005, thresh_above=0.05, to_plot=False, title="Waveform"):
    '''Returns the wavelenght (time) and if the esstimation is good (1=good, 0=uncertain) for an epochs timeseries.
    Can plot if desired.'''    
    time_start_per_antenna = get_impact_time(times, DATA, thresh_below=thresh_below, thresh_above=thresh_above)
    time_end_per_antenna = get_impact_time(times, DATA, thresh_below=thresh_below, thresh_above=thresh_above, reverse=True)
    if to_plot:
        fig, axes = tripple_plot(times, DATA, to_show=False, title=title)
        for i, ax in enumerate(axes):
            ax.axvline(time_start_per_antenna[i], linestyle='--', color='g')
            ax.axvline(time_end_per_antenna[i], linestyle='--', color='r')
        plt.show()
    good_est = 1    
    for t in time_end_per_antenna:
        if t - np.mean(time_start_per_antenna) > 10:
            good_est = 0
    return [t_end-t0 for t0, t_end in zip(time_start_per_antenna, time_end_per_antenna)], good_est