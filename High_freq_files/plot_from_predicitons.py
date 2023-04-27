import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import datetime
import math

from data_handling import smooth, get_data

folder_predictions = 'predictions'

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

solo_min_radius = [datetime.datetime(2020, 6, 15, 0, 0), datetime.datetime(2021, 2, 10, 0, 0), 
                   datetime.datetime(2021, 9, 9, 0, 0), datetime.datetime(2022, 3, 27, 0, 0), 
                   datetime.datetime(2022, 10, 12, 0, 0)]                                       #the dates of perihelion

# Apply a smoothening filter to smooth the signal
window_size = 30
model_smoothed = smooth(impact_model_count, window_size)
TDS_smoothed = smooth(impact_TDS_count, window_size)

plt.plot(dates, impact_model_count, '.', color = 'blue', label = 'Model classifications')
plt.plot(dates, model_smoothed, color='blue', label = 'Model smoothed')

plt.plot(dates, impact_TDS_count, '.', color = 'red', label = 'TDS classifications')
plt.plot(dates, TDS_smoothed, color='red', label = 'TDS smoothed')

plt.vlines(x = solo_min_radius, colors = 'black', linestyle="dashed", ymin = 0, ymax = max(impact_model_count), label = 'Solar orbiter at perihelion')

plt.xlabel('Dates', fontweight ='bold', fontsize = 25)
plt.xticks(fontsize = 25)

plt.ylabel('Impacts [/day]', fontweight ='bold', fontsize = 25)
plt.title('Predictions of dust impacts', fontsize = 30)

plt.grid(color='grey', linewidth=0.2)
plt.legend(fontsize="20")

plt.show()