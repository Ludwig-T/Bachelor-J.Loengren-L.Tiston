import os
import numpy as np
import tensorflow as tf
from data_handling import get_data, gen_timeseries, tripple_plot
from pre_processing import process_data
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Read Data
FILE_PATH = "C:/Data/High Freq/"
model_FILE_PATH = "C:\Data\model\model_run_GitHub"
model = tf.keras.models.load_model(model_FILE_PATH)

has_run = False
FLAGGED_EPOCHS = {}
for root, dirs, files in os.walk(FILE_PATH):    #iterate folders
    for file in files:                          #iterate files in folders
        if has_run:
            break
        if 'tswf' in file:                      #only use tswf-data
            filepath = os.path.join(root, file)
            print(filepath)
            WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO=False)
            flags = []
            for EPOCH in EPOCHS:
                times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                times_processed, DATA_processed = process_data(times, data, compression=4)
                DATA_processed = np.array(DATA_processed).transpose()
                DATA_processed = np.reshape(DATA_processed, (-1, 4096, 3))
                flags.append(model.predict(DATA_processed))
            FLAGGED_EPOCHS[file] = flags
            has_run = True
                
probabilities = FLAGGED_EPOCHS['solo_L2_rpw-tds-surv-tswf-e-cdag_20200601_V15.cdf']
flags = [item[0][1] if item[0][0] > 0.5 else item[0][1] for item in probabilities]
for EPOCH in EPOCHS:
    times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
    times_processed, DATA_processed = process_data(times, data, compression=4)
    tripple_plot(times, data, title=flags[EPOCH])
    #tripple_plot(times_processed, DATA_processed, title=flags[EPOCH])            
                