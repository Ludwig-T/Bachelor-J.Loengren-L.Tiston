import os
import numpy as np
import tensorflow as tf
from data_handling import get_data, gen_timeseries
from pre_processing import process_data
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Read Data
PATH = '//NAS24/solo/remote/data/L2/tds_wf_e'
YEAR = '2020'
MONTH = 1

FILE_PATH = f"{PATH}/{YEAR}"


model_FILE_PATH = 'C:/Users/joarl/OneDrive/Dokument/Skola/Kand/Kod_Kvammen/AndreasKvammen-ML_dust_detection-9c0d0de/CDF file classification/model_run_GitHub'
model = tf.keras.models.load_model(model_FILE_PATH)

FLAGGED_EPOCHS = {}
for root, dirs, files in os.walk(FILE_PATH):    #iterate folders
    for file in files:                          #iterate files in folders
        if 'tswf' in file:                      #only use tswf-data
            filepath = os.path.join(root, file)
            WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO=False)
            flags = []
            for EPOCH in EPOCHS:
                times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                times_processed, DATA_processed = process_data(times, data, compression=4)
                DATA_processed = np.array(DATA_processed).transpose()
                DATA_processed = np.reshape(DATA_processed, (-1, 4096, 3))
                flags.append(model.predict(DATA_processed))
            FLAGGED_EPOCHS[file] = flags
                
#probabilities = FLAGGED_EPOCHS['solo_L2_rpw-tds-surv-tswf-e-cdag_20200601_V15.cdf']
#flags = [item[0][1] if item[0][0] > 0.5 else item[0][1] for item in probabilities]