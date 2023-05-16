import cdflib
import numpy as np
import scipy.io as sio
import random

random.seed(42)

#window = 15360 creates a maximum of 1 minute for the maximum 256 SR
window = 512 #creates a maximum of 2 seconds snapshot

def get_data(filepath):
    '''Input: filepath of cdf-file with data
    '''
    #Read data
    c = cdflib.cdfread.CDF(filepath)
    V = c['V']
    E = c['E']
    EPOCH = c['EPOCH']
    
    return E, V, EPOCH


def pre_process(data):
    
    if isinstance(data, list):
        return_list = []

        for dates, keys, values in (data):

            wave_median = np.median(values, axis=2, keepdims=True)
            wave_nobias = values - wave_median
            mx = np.max(abs(wave_nobias))

            wave_processed = wave_nobias/(mx)

            return_list.append((dates, keys, wave_processed))
        
        return return_list


    else:
        #remove ofset
        data_median = np.median(data)
        data_nobias = data - data_median

    return data_nobias


def extracting_mat(PATH_MAT_FILE):
    return_list = []

    mat = sio.loadmat(PATH_MAT_FILE)
    l1 = mat['l1']
    fitparam = l1[0][0][4]

    t = l1[0][0][0]
    data = l1[0][0][1]

    if len(fitparam[0]) > 1:

        for r in range(fitparam.shape[0]):
            for c in range(fitparam.shape[1]):

                struc = fitparam[r, c]

                if len(struc[0][0]) > 9:
                    reference = struc[0][0][10]

                    if reference == 1:
                        
                        if len(struc[0][0]) > 11:
                            matlab_datenum = (struc[0][0][12][0][0])
                            return_list.append((c, matlab_datenum))

                        else:
                            print('Wrong length of structure?')


               
                else:
                    print('No ref found')

    return t, data, return_list


def sliding_data(data, window_size = window):
    
    # Define the overlap
    overlap = 0.5 #50%

    # Calculate the stride
    stride = int(window_size * overlap)

    start_indices = list(range(0, len(data) - window_size + 1, stride))



    return start_indices


def creating_indices(time, data, timestamps, window_size = window):
    
    # Define the overlap
    overlap = 0.5 #50%

    # Calculate the stride
    stride = int(window_size * overlap)

    start_indices = list(range(0, len(data) - window_size, stride))

    dust_stamps = []
    no_dust_stamps = []

    for window_start in start_indices:

        window_end = window_start + window_size

        if window_end + 1 > len(time):
            print('end time outside the dataset')

        else:
            
            time_start = time[window_start]
            time_end = time[window_end]

            for stamp in timestamps:

                if time_start <= stamp <= time_end:
                    dust_stamps.append(window_start)

    while len(no_dust_stamps) < len(dust_stamps[1::2]): #add equal ammount of dust

        random_sample = random.sample(start_indices, 1)

        if random_sample not in dust_stamps:
            
            no_dust_stamps.append(random_sample[0])
    
    return dust_stamps[1::2], no_dust_stamps #only return every other dust cus it is duplicated, here we chose the second one, so the dust start in the first half



def creating_timeseries(V, E, start_i_dust, start_i_no_dust, window_size = window):
    
    data_series_list = []
    labels_list = []

    for window_start in start_i_dust:

        window_end = window_start + window_size

        V_out = np.array([V[window_start:window_end]])
        E1_out = np.array([E[window_start:window_end, 0]])
        E2_out = np.array([E[window_start:window_end, 1]])
        data_out = np.array([V_out, E1_out, E2_out])

        data_series_list.append(data_out)
        labels_list.append(1)

    for window_start_0 in start_i_no_dust:

        window_end_0 = window_start_0 + window_size

        V_out_0 = np.array([V[window_start_0:window_end_0]])
        E1_out_0 = np.array([E[window_start_0:window_end_0, 0]])
        E2_out_0 = np.array([E[window_start_0:window_end_0, 1]])
        data_out_0 = np.array([V_out_0, E1_out_0, E2_out_0])

        data_series_list.append(data_out_0)
        labels_list.append(0)

    return data_series_list, labels_list