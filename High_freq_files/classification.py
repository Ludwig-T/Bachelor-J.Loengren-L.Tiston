import os
import numpy as np
import tensorflow as tf
from data_handling import get_data, gen_timeseries, tripple_plot
from pre_processing import process_data
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Read Data
FILE_PATH = "C:/Data/High Freq/10"
model_FILE_PATH = "C:\Data\model\model_run_GitHub"
model = tf.keras.models.load_model(model_FILE_PATH)

file_to_plot = 1

flag_counter = 0
has_run = False
file_counter = 1
FLAGGED_EPOCHS = {}
for root, dirs, files in os.walk(FILE_PATH):    #iterate folders
    for file in files:                          #iterate files in folders
        if has_run:
            break
        if 'tswf' in file:                      #only use tswf-data
            filepath = os.path.join(root, file)
            print(filepath)
            WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO=False)
            #flags = []
            for EPOCH in EPOCHS:
                times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                
                times_processed, DATA_processed = \
                    process_data(times, data, SAMPLING_RATE[EPOCH], compression=4, adaptable_noise=True)

                if DATA_processed is not None and file_counter == file_to_plot:
                    prediction = model.predict(DATA_processed)[0][1]
                    #flags.append(prediction)
                    if prediction > 0.95:
                        flag_counter += 1
                        process_data(times, data, SAMPLING_RATE[EPOCH], to_plot=True)
                    times_processed2, DATA_processed2 =\
                        process_data(times, data, SAMPLING_RATE[EPOCH], compression=4, adaptable_noise=True)
                    #tripple_plot(times, data, title=f"Prediction: {prediction:.2f}")
                    #tripple_plot(times_processed, DATA_processed, title=f"Prediction: {prediction:.2f}, static noise")
                    prediction = model.predict(DATA_processed2)[0][1]
                    #tripple_plot(times_processed2, DATA_processed2, title=f"Prediction: {prediction:.2f}, adaptable noise")  
            #FLAGGED_EPOCHS[file] = flags
            #file_counter += 1
                
"""probabilities = next(iter(FLAGGED_EPOCHS.values()))
flags = [item[0][1] if item[0][0] > 0.5 else item[0][1] for item in probabilities]
for EPOCH in EPOCHS:
    if EPOCH == 4:
        times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
        times_processed, DATA_processed = process_data(times, data, compression=4)
    if DATA_processed is not None:
        tripple_plot(times, data, title=flags[EPOCH])
        tripple_plot(times_processed, DATA_processed, title=flags[EPOCH])  """  
print(flag_counter)      
                
#41

#21