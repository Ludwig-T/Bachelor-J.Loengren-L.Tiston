import scipy.signal as scisig
import numpy as np
#import os
from data_handling import tripple_plot
import matplotlib.pyplot as plt

def process_data(times, DATA, SAMPLING_RATE=1, compression=1, to_plot=False):
    '''Removes bias, filters, compresses and normalizes data'''
    #Compress time
    if SAMPLING_RATE < 300000 and len(times) > 32000:
        print(f'Anomalous data: Sampling Rate = {SAMPLING_RATE}, Length = {len(times)}')
        print('Returning None')
        return None, None
    #Handles instances with dubble sampling freq. and timeseries lenght
    elif SAMPLING_RATE > 300000 and len(times) > 32000:
        print(f'Anomalous data: Sampling Rate = {SAMPLING_RATE}, Length = {len(times)}')
        print('Increasing compression')
        compression *= 2
    
    times_processed = np.linspace(times[0],times[-1],round(len(times)/compression))
    
    #Iterate over all three antennas
    DATA_processed = []
    for data in DATA:
        #Flatten and calculate median to remove bias
        F = 21
        flattened_data = scisig.medfilt(data, kernel_size=F)
        data_median = np.median(flattened_data)
        data_nobias = data - data_median
        
        #Filter high freq. noise
        filtered_data = scisig.medfilt(data_nobias, kernel_size=7)
        
        #Compress data
        data_compressed = np.interp(times_processed, times, filtered_data)
        
        #Normalize with respect to maximum
        mx = max(abs(data_compressed))
        data_processed = data_compressed/mx

        DATA_processed.append(data_processed)
        
        if to_plot:
            plt.plot(times, data_nobias)
            plt.title('No Bias')
            plt.show()

            plt.plot(times, filtered_data)
            plt.title('Filtered')
            plt.show()

            plt.plot(times_processed, data_compressed)
            plt.title('Compressed')
            plt.show()

            plt.plot(times_processed, data_processed)
            plt.title('Normalized')
            plt.show()
            
    #Reshape for model
    DATA_processed = np.array(DATA_processed).transpose()
    DATA_processed = np.reshape(DATA_processed, (1, 4096, 3))

    return times_processed, DATA_processed        


"""FILE_PATH = "C:/Data/High Freq/"
for root, dirs, files in os.walk(FILE_PATH):    #iterate folders
    for file in files:                          #iterate files in folders
        if 'tswf' in file:                      #only use tswf-data
            filepath = os.path.join(root, file)
            print(filepath)
            WAVEFORM, SAMPLING_RATE, FLAGGED_EPOCHS = get_data(filepath)
            for EPOCH in FLAGGED_EPOCHS:
                times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                times_processed, DATA_processed = process_data(times, data, compression=8)"""
       
            
            