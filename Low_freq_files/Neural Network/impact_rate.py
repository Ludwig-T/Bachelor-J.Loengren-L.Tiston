import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
l1_dir = '//NAS24/solo/remote/data/L1'
l1_pred_dir = 'C:/Githubs/kandidat/Low_freq_files/Neural Network/predictions'
l2_pred_dir = 'C:/Githubs/kandidat/High_freq_files/predictions'
uptime_dir = 'C:/Githubs/kandidat/Low_freq_files/uptime/saved_uptimes'

l1_files = []
l2_files = []
uptime_files = []
for filename in os.listdir(l1_pred_dir):
    l1_files.append(os.path.join(l1_pred_dir, filename))
    
for filename in os.listdir(l2_pred_dir):
    l2_files.append(os.path.join(l2_pred_dir, filename))

for filename in os.listdir(uptime_dir):
        uptime_files.append(os.path.join(uptime_dir, filename))
        
l1_data = [pd.read_pickle(file) for file in l1_files]
l2_data = [pd.read_csv(file) for file in l2_files]
uptime_data = [pickle.load(open(file, "rb")) for file in uptime_files]

print(uptime_data)
dates = []
impact_count = []
impact_rate = []
for file, uptime in zip(l1_data, uptime_data):
    for day in file:
        date = datetime.strptime(day, "%Y%m%d")
        dates.append(date)
        impact_count.append(np.count_nonzero(~np.isnan(file[day])))
        impact_rate.append(impact_count[-1]/uptime[date])
                
plt.plot(dates, impact_count, '.')
plt.plot(dates, impact_rate)
plt.show()