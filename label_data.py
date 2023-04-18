import pandas as pd
import os
from data_handling import get_data, gen_timeseries
from get_amp import get_amp
from get_wavelenght import get_wavelenght

filepath_pred = 'C:/Data/predictions/predictions_2020_05.csv'
df = pd.read_csv(filepath_pred)

folderpath_data = f'//NAS24/solo/remote/data/L2/tds_wf_e/{filepath_pred[-11:-7]}/{filepath_pred[-6:-4]}'
#Threshold for prediction
threshold = 0.9

epochs_info = []
dic = {}

for file in df.columns:
    filepath = os.path.join(folderpath_data, file)
    WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO=False)
    QUALITY, _, _ = get_data(filepath, to_extract='QUALITY_FACT')
    for EPOCH, prediction in enumerate(df[file]):
        if prediction > threshold:
            X, Y = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
            amplitude = get_amp(X, Y, to_plot=False)
            amplitude = [abs(Mx - mn) for mn, Mx in amplitude]
            wavelenght = get_wavelenght(X, Y, to_plot=False, title=str(QUALITY[EPOCH]), thresh_above=0.04, thresh_below=0.001)
            epochs_info.append([EPOCH, QUALITY[EPOCH], prediction, amplitude, wavelenght])
dic[file] = epochs_info
new_df = pd.DataFrame(dic)
new_df.to_csv('Labels.csv', index=False, header=True)
  
       
       