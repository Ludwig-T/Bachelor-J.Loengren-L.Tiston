import os
import pickle
from data_handling_L1 import get_data, extracting_mat, sliding_data, creating_indices, creating_timeseries

'''
Code for importing data from Matlab into Python and forming training data
'''
PATH = '//NAS24/solo/remote/data/L1'
PATH_FOLDER_MAT_FILES = '//NAS24/solo/data_michiko/dustdata'
save_as = 'data_files_2s.pkl'

main_list = []

B = False

for root, dirs, files in os.walk(PATH):    #iterate folders
    for file in files:
        if 'rpw-lfr-surv-cwf-cdag' in file:

            PATH_MAT_FILE = f'{PATH_FOLDER_MAT_FILES}/dustdata_{file[-16:-8]}.mat'

            if os.path.exists(PATH_MAT_FILE):

                print(PATH_MAT_FILE)

                CURRENT_PATH = f'{PATH}/{file[-16:-12]}/{file[-12:-10]}/{file[-10:-8]}/{file}'

                E, V, EPOCH  = get_data(CURRENT_PATH) 

                time, data, timestamps_tuple = extracting_mat(PATH_MAT_FILE)
                start_indices = sliding_data(time, overlap = 0.5)

                timestamps = [x[1] for x in timestamps_tuple]

                dust_indices, no_dust_indices = creating_indices(time, V, timestamps)

                data_series_list, labels_list = creating_timeseries(V, E, dust_indices, no_dust_indices)

                for i, data in (zip(labels_list, data_series_list)):
                    main_list.append((f'{file[-16:-12]}_{file[-12:-10]}_{file[-10:-8]}',i, data)) 
    
        if file == "solo_L1_rpw-lfr-surv-cwf-cdag_20220301_V04.cdf":
            B = True
            break

    if B:
        break
    
with open(save_as, 'wb') as f:
    pickle.dump(main_list, f)
