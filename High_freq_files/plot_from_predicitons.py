import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os
import datetime
import math
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from data_handling import get_data

folder_predictions = 'Test_predictions'

path_predictions = Path(__file__).with_name(folder_predictions) 

threshold = 0.9

impact_model_count = []
impact_TDS_count = []
dates = []

for file in os.listdir(path_predictions): 

    print(file) #to see it updating

    file_path = f'{path_predictions}/{file}'
    path_string = str(file)

    folderpath_data = f'//NAS24/solo/remote/data/L2/tds_wf_e/{path_string[-11:-7]}/{path_string[-6:-4]}' #extracting year and month so to get to right folder
    df = pd.read_csv(file_path)

    for column in df.columns:
        Model_EPOCHS = []
        filepath = os.path.join(folderpath_data, column)
        WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO=False)
        WAVEFORM_flag, SAMPLING_RATE_flag, FLAGGED_EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO=True)

        for EPOCH, prediction in enumerate(df[column]):

            if prediction is None:
                pass

            elif math.isnan(prediction):
                break

            elif prediction > threshold:
                Model_EPOCHS.append(prediction)     

        impact_model_count.append(len(Model_EPOCHS))
        impact_TDS_count.append(len(FLAGGED_EPOCHS))
        dates.append(column[-16:-8])


dates = [datetime.datetime.strptime(date, '%Y%m%d') for date in dates] #convert from str to dates

solo_min_radius = [datetime.datetime(2021, 2, 10, 0, 0), #datetime.datetime(2020, 6, 15, 0, 0)
                   datetime.datetime(2021, 9, 9, 0, 0), datetime.datetime(2022, 3, 27, 0, 0), 
                   datetime.datetime(2022, 10, 12, 0, 0)] #the dates of perihelion


"""num_dates = mdates.date2num(dates)
"""


# Apply the Savitzky-Golay filter to smooth the signal
window_size = 20
poly_order = 2
model_smoothed = savgol_filter(impact_model_count, 40, 5)
TDS_smoothed = savgol_filter(impact_TDS_count, 40, 5)


"""window = 25

average_model = []
average_TDS = []

for i in range(len(impact_model_count)):       #  - window + 1
    average_model.append(np.mean(impact_model_count[i:i+window]))
    average_TDS.append(np.mean(impact_TDS_count[i:i+window]))

#for i in range (window - 1):
#    average_model.insert(0, np.nan)
#    average_TDS.insert(0, np.nan)


# plot trend line
plt.plot(dates, average_model, color='blue', label='Average Model')
plt.plot(dates, average_TDS, color='red', label='Average TDS')"""


#apply medfilt

plt.plot(dates, model_smoothed, color='blue', label = 'Model Smoothed')
plt.plot(dates, TDS_smoothed, color='red', label = 'TDS Smoothed')

plt.plot(dates, impact_model_count, '.', color = 'blue', label = 'Model')
plt.plot(dates, impact_TDS_count, '.', color = 'red', label = 'TDS')

plt.vlines(x = solo_min_radius, colors = 'black', linestyle="dashed", ymin = 0, ymax = max(impact_model_count), label = 'Solar orbiter at perihelion')

plt.xlabel('Dates')
plt.ylabel('Known impacts [/day]')
plt.legend()

plt.show()


#medfilt, Gaussian keffa