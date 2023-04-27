import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
from numpy.polynomial.polynomial import Polynomial
import scipy.fftpack as scifft

def get_radius(date_str_list):
    """
    Takes a list of dates in format 'YYYYMMDD'
    and returns list of the averaged radius per day of SOLO to the sun
    in Astronomical units
    """
    df = pd.read_csv(path, sep=' ')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
    daily_mean = df.groupby('datetime')['R[AU]'].mean()
    return [daily_mean[date_str] for date_str in date_str_list]

def plot_hist(data):
    amplitudes = []
    wavelenghts = []
    for year in data:
        for day in year:
            print(day)
            for amplitude, wavelenght, waveQ in zip(year[day]['amplitude'], year[day]['wavelenght'], year[day]['waveQ']):
                amplitudes.append(max(amplitude))
                good_wavel = np.array(wavelenght)[np.array(waveQ, dtype=bool)]
                if len(good_wavel) != 0:
                    wavelenghts.append(max(good_wavel))
                    
    plt.hist(amplitudes, 20)
    plt.xlabel('Amplitude (V)')
    plt.show()
    
    plt.hist(wavelenghts, 20)
    plt.xlabel('Wavelength (ms)')
    plt.show()

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def fft(y):
    N = len(y)
    w = scifft.rfft(y)
    f = scifft.rfftfreq(N)
    spectrum = w**2
    
    cutoff_idx = spectrum < (spectrum.max()/5)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    y2 = scifft.irfft(w2)
    return y2
                
def plot_amp_time(data):
    
    daily_data = []
    dates = []
    date_str_list = []
    solo_dates = []
    for year in data:
        for day in year:
            amplitudes = []
            date_str = day.split('_')[3]  # Extract the date string
            date_str_list.append(date_str)
            date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            solo_dates.append(date)
            for amplitude in year[day]['amplitude']:
                amplitudes.append(max(amplitude))
            
            if len(amplitudes) != 0:
                daily_data.append(sum(amplitudes)/len(amplitudes))
                dates.append(date)
        
    solo_radi = get_radius(date_str_list)
    data_filtered = smooth(daily_data, 50)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(dates, daily_data, '.', label='Average max amplitude per day', color='blue', alpha=0.3)
    ax1.plot(dates, data_filtered, label='Interpolated curve', color='blue')
    ax1.set_ylabel('Amplitude (V)')
    ax2.plot(solo_dates, solo_radi, label='Distance from sun', linestyle='--', color='black', alpha=0.9)
    ax2.set_ylabel('Distance (A.U.)')
    ax2.set_ylim(0, 1.1)
    fig.suptitle('Max amplitude averaged per day')
    fig.legend()
    plt.grid(alpha=0.2)
    plt.show()

def plot_amp_space(data):
    
    daily_data = []
    dates = []
    date_str_list = []
    solo_dates = []
    for year in data:
        for day in year:
            amplitudes = []
            date_str = day.split('_')[3]  # Extract the date string
            for amplitude in year[day]['amplitude']:
                amplitudes.append(max(amplitude))
            
            if len(amplitudes) != 0:
                daily_data.append(sum(amplitudes)/len(amplitudes))
                date_str_list.append(date_str)
                
    solo_radi = get_radius(date_str_list)
    m, k = Polynomial.fit(solo_radi, daily_data, 1).coef
    plt.plot(solo_radi, daily_data, '.', alpha=0.8)
    #plt.axline((0, m), slope=k, color='orange')
    plt.xlim((0.28, 1.05))
    plt.xlabel('Distance from sun (A.U.)')
    plt.ylabel('Amplitude (V)')
    plt.title('Amplitude of dust impacts')
    plt.show()
    
def plot_wave_time(data):
    
    daily_data = []
    dates = []
    date_str_list = []
    solo_dates = []
    for year in data:
        for day in year:
            wavelenghts = []
            date_str = day.split('_')[3]  # Extract the date string
            date_str_list.append(date_str)
            date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            solo_dates.append(date)
            for wavelenght, waveQ in zip(year[day]['wavelenght'], year[day]['waveQ']):
                good_wavel = np.array(wavelenght)[np.array(waveQ, dtype=bool)]
                if len(good_wavel) != 0:
                    wavelenghts.append(max(good_wavel))
            
            if len(wavelenghts) != 0:
                daily_data.append(sum(wavelenghts)/len(wavelenghts))
                dates.append(date)
        
    solo_radi = get_radius(date_str_list)
    data_filtered = smooth(daily_data, 50)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(dates, daily_data, '.', label='Average max wavelength per day', color='blue', alpha=0.3)
    ax1.plot(dates[25:-25], data_filtered[25:-25], label='Interpolated curve', color='blue')
    ax1.set_ylabel('Wavelength (ms)')
    ax1.set_ylim(0, 16)
    ax2.plot(solo_dates, solo_radi, label='Distance from sun', color='black', alpha=0.9)
    ax2.set_ylabel('Distance (A.U.)')
    ax2.set_ylim(0, 1.1)
    fig.suptitle('Average max wavelength per day')
    fig.legend()
    plt.grid(alpha=0.2)
    plt.show()

path = 'C:\Githubs\kandidat\SOLO_orbit\SOLO_orbit_HCI.txt'

files = ['Labels_2020.pkl', 'Labels_2021.pkl', 'Labels_2022.pkl', 'Labels_2023.pkl']
data = [pd.read_pickle(year) for year in files]          

plot_wave_time(data)