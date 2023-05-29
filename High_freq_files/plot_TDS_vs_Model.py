import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from data_handling import get_data

"""
Plots differences in TDS and neural network data
"""

folder_predictions = 'predictions'
path = Path(__file__).with_name(folder_predictions) 

threshold = 0.9     #choosing a threshold for the model to classify as dust
start_month = '2020-04'
end_month = '2023-04' #one more then the final month for some reason

#creating the counting lists
agree_m = []
TDS_m = []
model_m = []

for file in os.listdir(path): 
    
    #creating the list for each months
    agree = []
    TDS = []
    model = []

    file_path = f'{path}/{file}'
    path_string = str(file)

    folderpath_data = f'//NAS24/solo/remote/data/L2/tds_wf_e/{path_string[-11:-7]}/{path_string[-6:-4]}' #extracting year and month so to get to right folder
    df = pd.read_csv(file_path)


    #going through all data for each month
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
        
    #add to monthly count, normalize by how many days looked at in month
    agree_m.append(len(agree)/df.shape[1])
    TDS_m.append(len(TDS)/df.shape[1])
    model_m.append(len(model)/df.shape[1])

    print(file)

#cerates dates
dates = pd.date_range(start=start_month, end=end_month, freq='M')
months = [timestamp.strftime('%Y-%m') for timestamp in dates]


#Set font sizes
plt.rc('font', size=28)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
# Set the font family to the LaTeX font family
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['figure.constrained_layout.use'] = True   

#plot the figure
fig, ax = plt.subplots()

barWidth_wide = 0.7
barWidth_small = 0.35

agree_bar = ax.bar(np.arange(len(agree_m)), agree_m, barWidth_wide, label ='Both classifying dust', zorder=3)
TDS_bar = ax.bar(np.arange(len(TDS_m)) - barWidth_small/2 , TDS_m, barWidth_small, bottom=agree_m, label='Only TDS classification', zorder=3)
model_bar = ax.bar(np.arange(len(model_m)) + barWidth_small/2 , model_m, barWidth_small, bottom=agree_m, label='Only Model classification', zorder=3)

ax.set_xticks(np.arange(len(agree_m)))
ax.set_xticklabels(months, rotation=90)

for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

ax.set_ylabel('Mean impacts [/day]')

ax.legend()
ax.grid(color = "grey", zorder=0)

plt.title('Agreeance and dissagreence in the models')
plt.show()