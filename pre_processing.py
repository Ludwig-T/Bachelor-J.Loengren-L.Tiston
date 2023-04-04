import scipy.signal as scisig
import numpy as np
#import os
#from data_handling import get_data, gen_timeseries

def process_data(times, DATA, compression=1):
    '''Removes bias, filters, compresses and normalizes data'''
    #compress time
    processed_times = np.linspace(times[0],times[-1],round(len(times)/compression))
    processed_DATA = []
    for data in DATA:
        #Flatten and calculate median to remove bias
        F = 21
        flattened_data = scisig.medfilt(data, kernel_size=F)
        data_median = np.median(flattened_data)
        nobias_data = data - data_median
        
        filtered_data = scisig.medfilt(nobias_data, kernel_size=7)  #Filter data
        data_compressed = np.interp(processed_times, times, filtered_data)  #Compress data
        #Normalize with respect to median
        mx = max(abs(data_compressed))
        data_processed = data_compressed/mx
        
        processed_DATA.append(data_processed)

    return processed_times, processed_DATA        


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
       
            
            