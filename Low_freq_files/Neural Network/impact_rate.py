import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
import spiceypy

'''
Plots impact rate from LF classifications
'''

l1_dir = '//NAS24/solo/remote/data/L1'
l1_pred_dir = 'C:/Githubs/kandidat/Low_freq_files/Neural Network/predictions'
l2_pred_dir = 'C:/Githubs/kandidat/High_freq_files/predictions'
uptime_dir = 'C:/Githubs/kandidat/Low_freq_files/uptime/saved_uptimes'

def smooth(data, window, weights=None):
    if weights is None:
        weights = np.ones(window) / window
    else:
        weights = weights / np.sum(weights)
    return np.convolve(data, weights, mode='same')


l1_files = []
uptime_files = []

for filename in os.listdir(l1_pred_dir):
    l1_files.append(os.path.join(l1_pred_dir, filename))

for filename in os.listdir(uptime_dir):
        uptime_files.append(os.path.join(uptime_dir, filename))
        
l1_data = [pd.read_pickle(file) for file in l1_files]

uptime_data = [pickle.load(open(file, "rb")) for file in uptime_files]

print(uptime_data)
dates = []
impact_count = []
impact_rate = []
for file, uptime in zip(l1_data, uptime_data):
    for day in file:
        date = datetime.strptime(day, "%Y%m%d")
        
        if uptime[date] > 0.01:
            dates.append(date)
            impact_count.append(np.count_nonzero(~np.isnan(file[day]))) 
            impact_rate.append(impact_count[-1]/uptime[date])
                    
        else:
            print(date)

#Load spicepy kernels
spiceypy.furnsh('C:/Data/solo_master/kernels/lsk/naif0012.tls')
directory = "C:/Data/solo_master/kernels/spk"
for filename in os.listdir(directory):
    if filename.endswith(".bsp"):
        full_path = os.path.join(directory, filename)
        spiceypy.furnsh(full_path)

#Get state of Solar Orbiter at day
SolO_radii = []
for date in dates:
    date_et = spiceypy.datetime2et(date)
    solo_state = spiceypy.spkgeo(targ=-144, et=date_et, ref='ECLIPJ2000', obs=10)[0]
    R = np.sqrt(np.sum([r**2 for r in solo_state[:3]]))*6.68458712*10**(-9)
    SolO_radii.append(R)

#Set font sizes
plt.rc('font', size=28)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
# Set the font family to the LaTeX font family
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['figure.constrained_layout.use'] = True

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(dates, impact_rate, '.', label='Impact rate')
smooth_curve = smooth(impact_rate, 30)
ax1.plot(dates[15:-15], smooth_curve[15:-15], label='Rolling mean n=30')
ax2.plot(dates, SolO_radii, '--', color='black', label='Spacecraft distance from sun')
ax2.set_ylim(0, 1)
plt.title('Impact rate from low frequency data')
plt.xlabel('Dates')
ax1.set_ylabel('Impact rate [/day]')
ax2.set_ylabel('Distance from sun [A.U.]')
fig.legend(framealpha=1, fontsize=18)
plt.grid(alpha=0.2)
plt.show()