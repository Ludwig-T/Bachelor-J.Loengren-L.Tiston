import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data_handling import get_data

folder_predictions = 'predictions'
path = Path(__file__).with_name(folder_predictions) 

threshold = 0.95
start_month = '2020-04'
end_month = '2023-03' #one more then the final month for some reason

agree_m = []
TDS_m = []
model_m = []

for file in os.listdir(path): 

    agree = []
    TDS = []
    model = []

    file_path = f'{path}/{file}'
    path_string = str(file)

    folderpath_data = f'//NAS24/solo/remote/data/L2/tds_wf_e/{path_string[-11:-7]}/{path_string[-6:-4]}' #extracting year and month so to get to right folder
    df = pd.read_csv(file_path)

    for column in df.columns:

        filepath = os.path.join(folderpath_data, column)
        WAVEFORM, SAMPLING_RATE, EPOCHS = get_data(filepath, to_extract='WAVEFORM_DATA', want_DOWNLINK_INFO=False)
        Q, H, FLAGGED_EPOCHS = get_data(filepath, to_extract='QUALITY_FACT', want_DOWNLINK_INFO=True)

        for EPOCH, prediction in enumerate(df[column]):

            if prediction > threshold and EPOCH in FLAGGED_EPOCHS:
                agree.append(prediction)
                
            elif prediction < threshold and EPOCH in FLAGGED_EPOCHS:
                TDS.append(prediction)

            elif prediction > threshold and EPOCH not in FLAGGED_EPOCHS:
                model.append(prediction)
        
    agree_m.append(len(agree))
    TDS_m.append(len(TDS))
    model_m.append(len(model))

    print(file)


dates = pd.date_range(start=start_month, end=end_month, freq='M')
months = [timestamp.strftime('%Y-%m') for timestamp in dates]
data = [agree_m, TDS_m, model_m]

barWidth = 0.25
fig = plt.subplots()
 
# Set position of bar on X axis
br1 = np.arange(len(data[0]))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, data[0], color ='r', width = barWidth,
        edgecolor ='grey', label ='Agree')
plt.bar(br2, data[1], color ='g', width = barWidth,
        edgecolor ='grey', label ='TDS')
plt.bar(br3, data[2], color ='b', width = barWidth,
        edgecolor ='grey', label ='Model')
 
# Adding Xticks
plt.xlabel('Month', fontweight ='bold', fontsize = 12)
plt.ylabel('Occurrence', fontweight ='bold', fontsize = 12)
plt.xticks([r + barWidth for r in range(len(data[0]))],
        months, rotation='vertical')
plt.title('Number of agreence in the models', fontweight ='bold', fontsize = 15)
plt.legend()
plt.show()