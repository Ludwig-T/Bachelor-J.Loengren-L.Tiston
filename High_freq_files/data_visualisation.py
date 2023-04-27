import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
from pathlib import Path

def get_radius(date_str_list):
    """
    Takes a date in format 'YYYY-MM-DD'
    and returns the averaged radius of SOLO to the sun
    in Astronomical units
    """
    df = pd.read_csv(path, sep=' ')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
    daily_mean = df.groupby('datetime')['R[AU]'].mean()
    return [daily_mean[date_str] for date_str in date_str_list]

def plot_hist(df):
    amplitudes = []
    wavelenghts = []

    for day in df:
        print(day)
        for amplitude, wavelenght, waveQ in zip(df[day]['amplitude'], df[day]['wavelenght'], df[day]['waveQ']):
            amplitudes.append(max(amplitude))
            good_wavel = np.array(wavelenght)[np.array(waveQ, dtype=bool)]
            if len(good_wavel) != 0:
                wavelenghts.append(max(good_wavel))

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
                
def plot_time(df):
    daily_data = []
    dates = []
    date_str_list = []
    solo_dates = []
    for day in df:
        amplitudes = []
        date_str = day.split('_')[3]  # Extract the date string
        date_str_list.append(date_str)
        date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        solo_dates.append(date)
        for amplitude in df[day]['amplitude']:
            amplitudes.append(max(amplitude))
        
        if len(amplitudes) != 0:
            daily_data.append(sum(amplitudes)/len(amplitudes))
            dates.append(date)
    
    solo_radi = get_radius(date_str_list)
    data_filtered = smooth(daily_data, 30)
    plt.plot(dates, daily_data, '.')
    plt.plot(dates, data_filtered)
    plt.plot(solo_dates, solo_radi)
    plt.show()

path = 'C:\Githubs\kandidat\SOLO_orbit\SOLO_orbit_HCI.txt'
df = pd.read_pickle('Labels_2022.pkl')          
plot_time(df)