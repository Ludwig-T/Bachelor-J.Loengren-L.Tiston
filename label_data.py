import pandas as pd
import os
from data_handling import get_data, gen_timeseries
from get_amp import get_amp
from get_wavelenght import get_wavelenght

save_as = 'Labels_2022.pkl'
#Folder with predictions:
folderpath_pred = 'C:/Data/predictions'

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
            WAVEFORM, SAMPLING_RATE, EPOCHS, FLAGs =\
                get_data(filepath, to_extract='WAVEFORM_DATA_VOLTAGE', want_DOWNLINK_INFO='Both')
            
            data_dict = {
            "epoch": [],        #Epoch index
            "downlink": [],     #Onboard computer flag for dust
            "prediction": [],   #Neural net prediction for dust
            "amplitude": [],    #Amplitude of signals
            "wavelenght": [],   #Wavelenght of signals
            "waveQ": []         #Quality of wavelenght essitmations
            }
            
            for EPOCH, prediction in enumerate(df[file]):
                if prediction > threshold:
                    X, Y = gen_timeseries(WAVEFORM, SAMPLING_RATE, EPOCH)
                    amplitude = get_amp(X, Y, to_plot=False)
                    amplitude = [abs(Mx - mn) for mn, Mx in amplitude]
                    wavelenght, waveQ = get_wavelenght(X, Y, to_plot=False,\
                        thresh_below=[0.01, 0.01, 0.01], thresh_above=[0.02, 0.02, 0.02])
                    
                    data_dict['epoch'].append(EPOCH)
                    data_dict['downlink'].append(FLAGs[EPOCH])
                    data_dict['amplitude'].append(amplitude)
                    data_dict['prediction'].append(prediction)
                    data_dict['wavelenght'].append(wavelenght)
                    data_dict['waveQ'].append(waveQ)

            main_dic[file] = data_dict
            print(str(file) + ' complete')
    else:
        print("Couldn't find file at {filepath_pred}.")

new_df = pd.DataFrame.from_dict(main_dic, orient='index')
new_df = new_df.transpose()
new_df.to_pickle(save_as)
  
       
       