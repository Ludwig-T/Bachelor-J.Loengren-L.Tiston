import os
import numpy as np
import tensorflow as tf
from data_handling import get_data, gen_timeseries
from pre_processing import process_data
import pandas as pd

#Read Data
PATH = '//NAS24/solo/remote/data/L2/tds_wf_e'
YEAR = '2023'
MONTH = '03' #need to be in format XX, not just X

FILE_PATH = f"{PATH}/{YEAR}/{MONTH}"
model_FILE_PATH = 'C:/Users/joarl/OneDrive/Dokument/Skola/Kand/Kod_Kvammen/AndreasKvammen-ML_dust_detection-9c0d0de/CDF file classification/model_run_GitHub'
model = tf.keras.models.load_model(model_FILE_PATH)

df = pd.DataFrame() #create dataframe
FLAGGED_EPOCHS = {}
DAY = 1 #day-counter

for file in os.listdir(FILE_PATH):    #iterate folder
    if 'tswf' in file:                #only use tswf-data
        if DAY%2:                     #take only odd days
            current_file_path = f"{FILE_PATH}/{file}"
            WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(current_file_path, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO=False)
            flags = []
            for EPOCH in EPOCHS:
                times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                times_processed, DATA_processed = process_data(times, data, compression=4)
                DATA_processed = np.array(DATA_processed).transpose()
                DATA_processed = np.reshape(DATA_processed, (-1, 4096, 3))
                flags.append(model.predict(DATA_processed))
            FLAGGED_EPOCHS[file] = flags
            probabilities = FLAGGED_EPOCHS[file]
            prob = [item[0][1] for item in probabilities] #save prob dust

            df_day = pd.DataFrame({f'{DAY}': prob})
            df = pd.concat([df, df_day], axis=1)
            DAY += 1

            print(DAY)
        else:
            DAY +=1

FILENAME = f"Predicitons_{YEAR}_{MONTH}.csv"
df.to_csv(FILENAME, index=False)