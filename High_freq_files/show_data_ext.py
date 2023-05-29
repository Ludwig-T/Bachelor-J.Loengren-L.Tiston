import os
import numpy as np
import tensorflow as tf
from data_handling import get_data, gen_timeseries, tripple_plot
from pre_processing import process_data
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
Code for showing different data extension options
"""

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#Set font sizes
plt.rc('font', size=28)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
# Set the font family to the LaTeX font family
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['figure.constrained_layout.use'] = True       
#Read Data
FILE_PATH = "C:/Data/anomaly/"
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
            for EPOCH in EPOCHS:
                DATA_processed = [None] * 4
                times_processed = [None] * 4
                preds = [None] * 4
                
                times, data = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                
                times_processed[0], DATA_processed[0] = \
                    process_data(times, data, SAMPLING_RATE[EPOCH], compression=4, extension='None')
                times_processed[1], DATA_processed[1] = \
                    process_data(times, data, SAMPLING_RATE[EPOCH], compression=4, extension='copy')
                times_processed[2], DATA_processed[2] = \
                    process_data(times, data, SAMPLING_RATE[EPOCH], compression=4, extension='noise', adaptable_noise=False)
                times_processed[3], DATA_processed[3] = \
                    process_data(times, data, SAMPLING_RATE[EPOCH], compression=4, extension='noise', adaptable_noise=True)
                
                if DATA_processed[0] is not None and file_counter == file_to_plot:
                    preds[3] = model.predict(DATA_processed[3])[0][1]
                    
                    preds[0] = model.predict(DATA_processed[0])[0][1]
                    if preds[3] > 0.95:# and preds[0] < 0.05:
                        preds[1] = model.predict(DATA_processed[1])[0][1]
                        preds[2] = model.predict(DATA_processed[2])[0][1]
                        
                        
                        fig, axs = plt.subplots(2, 2, layout='constrained')
                        axs = axs.flatten()
                        titles = ['Not extended', 'Copy of signal', 'Static noise', 'Adaptable noise']
                        for i, (t, y, p) in enumerate(zip(times_processed, DATA_processed, preds)):
                            axs[i].plot(t, y[0,:,0])
                            axs[i].set_title(f"{titles[i]}. Prediction: {p:.2f}")
                        
                        fig.supxlabel('Time [ms]')
                        fig.supylabel('Amplitude (Normalized)')
                        plt.show()