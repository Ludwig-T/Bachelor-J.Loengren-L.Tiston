from data_handling import tripple_plot
import scipy.signal as scisig
import numpy as np
import matplotlib.pyplot as plt

def get_impact_time(times, DATA, reverse=False, \
        THRESH_HIGH=[0.05, 0.05, 0.05], THRESH_LOW=[0.005, 0.005, 0.005]):
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

    for thresh_above, thresh_below, data in zip(THRESH_HIGH, THRESH_LOW, DATA):
        #Process data:
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
                
        
def get_wavelenght(times, DATA, \
    thresh_below, thresh_above, to_plot=False, title="Quality"):
    '''Returns the wavelenght (time) and if the esstimation is good (1=good, 0=uncertain) for an epochs timeseries.
    Can plot if desired.'''
    
    max_indices = [np.argmax(np.abs(subarray)) for subarray in DATA]
    t_lim10 = times[len(times)//10]
    good_est = 1
    
    done = False
    while not done:  
        time_start_per_antenna = get_impact_time(times, DATA, THRESH_HIGH=thresh_above, THRESH_LOW=thresh_below)
        time_end_per_antenna = get_impact_time(times, DATA, THRESH_HIGH=thresh_above, THRESH_LOW=thresh_below,\
            reverse=True)

        checks = 0
        for i, (t_start, t_end) in enumerate(zip(time_start_per_antenna, time_end_per_antenna)):
            time_max = times[max_indices[i]]
            #If t_start is too far from the max peak or the wavelenght is too long, it is probably detecting noise.
            if abs(time_max - t_start) > t_lim10/2 or abs(t_end - t_start > 10):
                #Increase parameter
                thresh_above[i] *= 1.8

                #Couldn't find good wavelenght
                if thresh_above[i] > 1:
                    good_est = 0
                    done = True
                    break
            else:
                checks += 1

        #All three antennas should be good.
        if checks == 3:
            break
    
    for t in time_end_per_antenna:
        #Wavelenghts shouldn't differ that much.
        if t - np.mean(time_start_per_antenna) > 10:
            good_est = 0

    if to_plot:
        if title == 'Quality':
            title = good_est
        fig, axes = tripple_plot(times, DATA, to_show=False, title=title)
        for i, ax in enumerate(axes):
            ax.axvline(time_start_per_antenna[i], linestyle='--', color='g')
            ax.axvline(time_end_per_antenna[i], linestyle='--', color='r')
        plt.show()
    
    return [t_end-t0 for t0, t_end in zip(time_start_per_antenna, time_end_per_antenna)], good_est