import pandas as pd
import os
from data_handling import get_data, gen_timeseries
from get_amp import get_amp
from get_wavelenght import get_wavelenght

#Folder with predictions:
folderpath_pred = 'C:/Data/predictions'
#filepath_pred = 'C:/Data/predictions/predictions_2020_05.csv'

#Threshold for prediction
threshold = 0.9
main_dic = {}

for filename in os.listdir(folderpath_pred):
    filepath_pred = os.path.join(folderpath_pred, filename)
    folderpath_data = f'//NAS24/solo/remote/data/L2/tds_wf_e/{filepath_pred[-11:-7]}/{filepath_pred[-6:-4]}'
    # checking if it is a file
    if os.path.isfile(filepath_pred):
        df = pd.read_csv(filepath_pred)

        for file in df.columns:
            filepath = os.path.join(folderpath_data, file)
            WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE',\
                                                        want_DOWNLINK_INFO=False)
            
            data_dict = {
            "epoch": [],
            "prediction": [],
            "amplitude": [],
            "wavelenght": [],
            "waveQ": []
            }
            
            for EPOCH, prediction in enumerate(df[file]):
                if prediction > threshold:
                    X, Y = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                    amplitude = get_amp(X, Y, to_plot=False)
                    amplitude = [abs(Mx - mn) for mn, Mx in amplitude]
                    wavelenght, waveQ = get_wavelenght(X, Y, to_plot=False,\
                        thresh_below=[0.01, 0.01, 0.01], thresh_above=[0.02, 0.02, 0.02])
                    
                    data_dict['epoch'].append(EPOCH)
                    data_dict['amplitude'].append(amplitude)
                    data_dict['prediction'].append(prediction)
                    data_dict['wavelength'].append(wavelenght)
                    data_dict['waveQ'].append(waveQ)

            main_dic[file] = data_dict
            print(str(file) + ' complete')

new_df = pd.DataFrame.from_dict(main_dic, orient='index')
new_df = new_df.transpose()
new_df.to_pickle('Labels.pkl')
  
       
       