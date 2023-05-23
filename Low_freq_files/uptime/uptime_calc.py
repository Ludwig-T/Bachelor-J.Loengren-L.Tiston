import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import spiceypy
import cdflib
import pickle
spiceypy.furnsh('C:/Data/solo_master/kernels/lsk/naif0012.tls')

l1_dir = '//NAS24/solo/remote/data/L1'
l1_pred_dir = 'C:/Githubs/kandidat/Low_freq_files/Neural Network/predictions'

save_as = 'uptime_202112.pkl'   #Change this
count_of_file = 1               #And this

l1_files = []

count = 0
for filename in os.listdir(l1_pred_dir):
    count += 1
    if count == count_of_file:
        print(filename)
        l1_files.append(os.path.join(l1_pred_dir, filename))
    
l1_data = [pd.read_pickle(file) for file in l1_files]

dates = []
impact_count = []
for file in l1_data:
    for day in file:
        dates.append(datetime.strptime(day, "%Y%m%d"))
        impact_count.append(np.count_nonzero(~np.isnan(file[day])))
downtime_threshold = timedelta(seconds=0.063*32)
time_day = timedelta(hours=24)

uptime_dic = {}
for date in dates:
    target_date_str = datetime.strftime(date, "%Y%m%d")    
    for root, dirs, files in os.walk(l1_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if target_date_str in file and 'cwf' in file:
                epochs = cdflib.cdfread.CDF(file_path)['Epoch'][::32]
                datetime_list = [spiceypy.et2datetime(et*10**(-9)) for et in epochs]
                total_duration = datetime_list[-1] - datetime_list[0]    
                downtime_duration = timedelta()
                for i in range(1, len(datetime_list)):
                    time_diff = datetime_list[i] - datetime_list[i-1]
                    if time_diff > downtime_threshold:
                        downtime_duration += time_diff
                        
                total_downtime = time_day - total_duration + downtime_duration
                uptime = (1 - total_downtime/time_day)       
                uptime_dic[date] = uptime
                print(uptime)
                
pickle.dump(uptime_dic, open(save_as, "wb"))
