import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from scipy.signal import savgol_filter
from numpy.polynomial.polynomial import Polynomial
import spiceypy
import os

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

def smooth(data, window, weights=None):
    if weights is None:
        weights = np.ones(window) / window
    else:
        weights = weights / np.sum(weights)
    return np.convolve(data, weights, mode='same')

def str2et(date_str):
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    date_et = spiceypy.utc2et(formatted_date)
    return date_et

def get_good_wave(wavelength, waveQ):
    good_wavel = np.array(wavelength)[np.array(waveQ, dtype=bool)]
    good_wavel = [wave for wave in good_wavel if wave < 15]
    return good_wavel

def plot_hist(data):
    amplitudes = []
    wavelengths = []
    for year in data:
        for day in year:
            print(day)
            for amplitude, wavelength, waveQ in zip(year[day]['amplitude'], year[day]['wavelength'], year[day]['waveQ']):
                amplitudes.append(max(amplitude))
                good_wavel = get_good_wave(wavelength, waveQ)
                if len(good_wavel) != 0:
                    wavelengths.append(max(good_wavel))
    
    fig, axs = plt.subplots(1, 2, layout='constrained')                
    axs[0].hist(amplitudes, 20)
    axs[0].set_title('Amplitude distrubition')
    axs[0].set_xlabel('Amplitude [V]')
    
    axs[1].hist(wavelengths, 20)
    axs[1].set_title('Wavelength distrubition')
    axs[1].set_xlabel('Wavelength [ms]')

    plt.show()
                
def plot_amp_time(data):
    
    daily_data = []
    dates = []
    date_str_list = []
    solo_dates = []
    weights = []
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
                weights.append(len(amplitudes))
                dates.append(date)
        
    solo_radi = get_radius(date_str_list)
    #data_filtered = weighted_rolling_average(daily_data, weights, 41)
    data_filtered = smooth(daily_data, 41)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(dates, daily_data, '.', label='Daily average max', color='blue', alpha=0.3)
    ax1.plot(dates[20:-20], data_filtered[20:-20], label='Rolling mean n=41', color='blue')
    ax1.set_ylabel('Amplitude [V]')
    ax2.plot(solo_dates, solo_radi, label='Distance from sun', linestyle='--', color='black', alpha=0.9)
    ax2.set_ylabel('Distance [A.U.]')
    ax2.set_ylim(0, 1.5)
    fig.suptitle('Average max amplitude per day')
    fig.legend(framealpha=1)
    plt.grid(alpha=0.2)
    plt.show()

def plot_amp_space(data):
    
    daily_data = []
    date_str_list = []
    total_data = []
    total_date_str = []
    for year in data:
        for day in year:
            amplitudes = []
            date_str = day.split('_')[3]  # Extract the date string
            for amplitude in year[day]['amplitude']:
                amplitudes.append(max(amplitude))
            if len(amplitudes) != 0:
                daily_data.append(sum(amplitudes)/len(amplitudes))
                date_str_list.append(date_str)
                total_data.append(amplitudes)
                total_date_str.append([date_str]*len(amplitudes))
                
    solo_radi = get_radius(date_str_list)
    solo_radi
    #m, k = Polynomial.fit(solo_radi, daily_data, 1).coef
    plt.plot(solo_radi, daily_data, '.', alpha=0.8)
    #plt.axline((0, m), slope=k, color='orange')
    plt.xlim((0.28, 1.05))
    plt.xlabel('Distance from sun (A.U.)')
    plt.ylabel('Amplitude (V)')
    plt.title('Average max amplitude per day')
    plt.grid(True, alpha=0.2)
    plt.show()
    
def plot_wave_time(data):
    
    daily_data = []
    dates = []
    date_str_list = []
    solo_dates = []
    for year in data:
        for day in year:
            wavelengths = []
            date_str = day.split('_')[3]  # Extract the date string
            date_str_list.append(date_str)
            date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            solo_dates.append(date)
            for wavelength, waveQ in zip(year[day]['wavelength'], year[day]['waveQ']):
                good_wavel = get_good_wave(wavelength, waveQ)
                if len(good_wavel) != 0:
                    wavelengths.append(max(good_wavel))
            
            if len(wavelengths) != 0:
                daily_data.append(sum(wavelengths)/len(wavelengths))
                dates.append(date)
        
    solo_radi = get_radius(date_str_list)
    data_filtered = smooth(daily_data, 41)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(dates, daily_data, '.', label='Daily Data', color='blue', alpha=0.3)
    ax1.plot(dates[25:-25], data_filtered[25:-25], label='Rolling mean n=41', color='blue')
    ax1.set_ylabel('Wavelength [ms]')
    ax1.set_ylim(0, 16)
    ax2.plot(solo_dates, solo_radi, label='Distance from sun', color='black', alpha=0.9)
    ax2.set_ylabel('Distance [A.U.]')
    ax2.set_ylim(0, 1.1)
    fig.suptitle('Average max wavelength per day')
    fig.legend(fontsize=22, framealpha=0.95)
    plt.grid(alpha=0.2)
    plt.show()
    
def plot_amp_space_year(data):
    
    
    for year in data:
        daily_data = []
        date_str_list = []
        for day in year:
            amplitudes = []
            date_str = day.split('_')[3]  # Extract the date string
            for amplitude in year[day]['amplitude']:
                amplitudes.append(max(amplitude))
            
            if len(amplitudes) != 0:
                daily_data.append(sum(amplitudes)/len(amplitudes))
                date_str_list.append(date_str)
        solo_radi = get_radius(date_str_list)
        plt.plot(solo_radi, daily_data, '.')
                
        m, k = Polynomial.fit(solo_radi, daily_data, 1).coef
    #plt.plot(solo_radi, daily_data, '.', alpha=0.8)
        plt.axline((0, m), slope=k)
    #plt.xlim((0.28, 1.05))
    plt.xlabel('Distance from sun [A.U.]')
    plt.ylabel('Amplitude [V]')
    plt.title('Average max amplitude per day')
    plt.show()

def plot_amp_vel(data):
    daily_data = []
    solo_vel_list = []
    daily_data_len = []
    for year in data:
        for day in year:
            amplitudes = []
            
            for amplitude in year[day]['amplitude']:
                amplitudes.append(max(amplitude))
            
            if len(amplitudes) != 0:
                daily_data.append(sum(amplitudes)/len(amplitudes))
                daily_data_len.append(len(amplitudes))
                
                date_str = day.split('_')[3]  # Extract the date string
                date_et = str2et(date_str)
                solo_state = spiceypy.spkgeo(targ=-144, et=date_et, ref='ECLIPJ2000', obs=10)[0]
                solo_vel = np.sqrt(np.sum([v**2 for v in solo_state[3:]]))
                solo_vel_list.append(solo_vel)
                                 
    sc = plt.scatter(solo_vel_list, daily_data, s=daily_data_len, alpha=0.8)
    plt.legend(*sc.legend_elements("sizes", num=4), title='Number of impacts')
    plt.xlabel('Total orbital speed w.r.t. sun [km/s]')
    plt.ylabel('Amplitude [V]')
    plt.title('Average max amplitude per day')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_amp_dir(data):
    pos_old = 0
    daily_data_away = []
    daily_data_to = []
    v_solo_away = []
    v_solo_to = []
    for year in data:
        for day in year:
            amplitudes = []
            
            for amplitude in year[day]['amplitude']:
                amplitudes.append(max(amplitude))
            
            if len(amplitudes) != 0:
                amp_mean = sum(amplitudes)/len(amplitudes)
                
                date_str = day.split('_')[3]  # Extract the date string
                date_et = str2et(date_str)
                solo_state = spiceypy.spkgeo(targ=-144, et=date_et, ref='ECLIPJ2000', obs=10)[0]
                solo_vel = np.sqrt(np.sum([v**2 for v in solo_state[3:]]))
                pos_new = np.sqrt(np.sum([r**2 for r in solo_state[:3]]))
                if pos_new > pos_old:
                    daily_data_away.append(amp_mean)
                    v_solo_away.append(solo_vel)
                else:
                    daily_data_to.append(amp_mean)
                    v_solo_to.append(solo_vel)
                pos_old = pos_new
                
    plt.plot(v_solo_away, daily_data_away, '.', alpha=0.8, label='Away from sun')
    plt.plot(v_solo_to, daily_data_to, '.', alpha=0.8, label='Towards sun')
    plt.legend()
    plt.xlabel('Total orbital speed [km/s]')
    plt.ylabel('Amplitude [V]')
    plt.title('Dots are average max amplitude per day')
    plt.show()

def plot_amp_radV(data):
    daily_data = []
    daily_data_size = []
    solo_vel_list = []
    for year in data:
        for day in year:
            amplitudes = []
            
            for amplitude in year[day]['amplitude']:
                amplitudes.append(max(amplitude))
            
            if len(amplitudes) != 0:
                daily_data.append(sum(amplitudes)/len(amplitudes))
                daily_data_size.append(len(amplitudes))
                
                #Get state of Solar Orbiter at day
                date_str = day.split('_')[3]
                date_et = str2et(date_str)
                solo_state = spiceypy.spkgeo(targ=-144, et=date_et, ref='ECLIPJ2000', obs=10)[0]
                
                #Calculate radial velocity
                vx, vy, vz = solo_state[3:]
                R = np.sqrt(np.sum([r**2 for r in solo_state[:3]]))
                x, y, z = solo_state[:3]
                
                #Unit vectors towards sun
                ux = -x/R
                uy = -y/R
                uz = -z/R
                
                #Dot product
                vrad = vx*ux + vy*uy + vz*uz
                solo_vel_list.append(vrad)
                    
    sc = plt.scatter(solo_vel_list, daily_data, s=daily_data_size, alpha=0.8)
    plt.xlabel('Radial velocity towards sun (km/s)')
    plt.ylabel('Amplitude (V)')
    plt.title('Average max amplitude per day')
    plt.legend(*sc.legend_elements("sizes", num=4), title='Number of impacts')
    plt.grid(True, alpha=0.5)
    plt.show()
    
def plot_ant_amp(data):
    labels = ['Antenna 1', 'Antenna 2', 'Antenna 3']
    data_ant = [0, 0, 0]
    amp_ant = [([], []), ([], []), ([], [])]

    for year in data:
        for day in year:
            for amplitude in year[day]['amplitude']:
                amp_max = max(amplitude)
                ind_max = amplitude.index(max(amplitude))
                data_ant[ind_max] += 1
                amp_ant[ind_max][0].append(amp_max)
                
                date_str = day.split('_')[3]
                date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                amp_ant[ind_max][1].append(date)
                
    #fig, ax = plt.subplots()
    #ax.pie(data_ant, labels=labels)
    #plt.show()
    fig, axes = plt.subplots(3, sharex=True, sharey=True)
    for ant, ax, label in zip(amp_ant, axes, labels):
        ax.plot(ant[1], np.log(ant[0]), '.', markersize=2)
        ax.set_title(label)
    plt.xlabel('Date')
    fig.supylabel('Log(Voltage) [log(V)]')
    fig.suptitle('Amplitude of dust impacts')
    plt.show()
    
def plot_ant_day(data):
    labels = ['Antenna 1', 'Antenna 2', 'Antenna 3']
    data_ant = [0, 0, 0]
    amp_ant = [([], []), ([], []), ([], [])]

    for year in data:
        for day in year:
            amp_ant_day = [([], []), ([], []), ([], [])]
            for amplitude in year[day]['amplitude']:
                amp_max = max(amplitude)
                ind_max = amplitude.index(max(amplitude))
                data_ant[ind_max] += 1
                amp_ant_day[ind_max][0].append(amp_max)
                
                date_str = day.split('_')[3]
                date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                amp_ant_day[ind_max][1].append(date)
            
            for i, antenna in enumerate(amp_ant_day):
                if len(antenna[0]) != 0:
                    amp_ant[i][0].append(np.mean(antenna[0]))
                    amp_ant[i][1].append(date)
    
    colors = ['yellowgreen', 'orange', 'cadetblue']     
    fig, ax = plt.subplots()
    ax.pie(data_ant, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title(f'Max amplitude by antenna (total detections: {sum(data_ant)})')
    plt.show()
    for color, ant, label in zip(colors, amp_ant, labels):
        plt.plot(ant[1], np.log(ant[0]), '.', markersize=2, color=color)
        plt.plot(ant[1], np.log(savgol_filter(ant[0], 40, 3)), color=color, label=label)
    plt.xlabel('Date')
    plt.ylabel('log(Voltage) [log(V)]')
    plt.title(f'Average max amplitude per day')
    plt.legend(title='Savgol filter (n=40)')
    plt.show()

def polarity(data):
    polarity = [[], [], []]
    dates = []
    for file in data:
        for day in file:            
            for ampInfo in file[day]['ampInfo']:
                for i, antenna in enumerate(ampInfo):   
                    Amin, Amax, Amed = antenna[:3]
                    if abs(Amin-Amed) > abs(Amax-Amed):
                        polarity[i].append(0)
                    else:
                        polarity[i].append(1)

                date_str = day.split('_')[3]  # Extract the date string
                date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                dates.append(date)
                
    n = 420
    labels = ['Antenna 1', 'Antenna 2', 'Antenna 3']
    line_styles = ['-', '--', '-.']
    for i, antenna in enumerate(polarity):
        plt.plot(dates[n//2:-n//2], smooth(antenna, 420)[n//2:-n//2], line_styles[i], label=labels[i], )
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Frequency of which the largest absolute amplitude is positive')
    plt.ylim((0, 1))
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

def polarity_pDay(data):
    polarity = [[], [], []]
    dates = []
    for file in data:
        for day in file:     
            pol_daily = [[], [], []]     
            for ampInfo in file[day]['ampInfo']:
                for i, Aantenna in enumerate(ampInfo): 
                    Amin, Amax, Amed = Aantenna[:3]
                    if abs(Amin-Amed) > abs(Amax-Amed):
                        pol_daily[i].append(0)
                    else:
                        pol_daily[i].append(1)
                    
            if len(pol_daily[0]) != 0:
                date_str = day.split('_')[3]  # Extract the date string
                date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                dates.append(date)
                for i, pol_antenna in enumerate(pol_daily):
                    polarity[i].append(np.mean(pol_antenna))
                
    n = 42
    labels = ['Antenna 1', 'Antenna 2', 'Antenna 3']
    line_styles = ['-', '--', '-.']
    for i, antenna in enumerate(polarity):
        plt.plot(dates[n//2:-n//2], smooth(antenna, n)[n//2:-n//2], line_styles[i], label=labels[i], )
    #plt.plot(dates[20:-20], pol_smooth[20:-20])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Frequency largest absolute amplitude is positive')
    plt.legend()
    plt.show()

def plot_dust_radV(data):
    daily_data = []
    solo_vel_list = []
    solo_R = []
    for year in data:
        for day in year:
            
            daily_impacts = 0
            for _ in year[day]['amplitude']:
                daily_impacts += 1
            
            daily_data.append(daily_impacts)   
            #Get state of Solar Orbiter at day
            date_str = day.split('_')[3]
            date_et = str2et(date_str)
            solo_state = spiceypy.spkgeo(targ=-144, et=date_et, ref='ECLIPJ2000', obs=10)[0]
            
            #Calculate radial velocity
            vx, vy, vz = solo_state[3:]
            R = np.sqrt(np.sum([r**2 for r in solo_state[:3]]))
            x, y, z = solo_state[:3]
            
            #Unit vectors towards sun
            ux = -x/R
            uy = -y/R
            uz = -z/R
            
            #Dot product
            vrad = vx*ux + vy*uy + vz*uz
            solo_vel_list.append(vrad)
            solo_R.append(R)
    x = []
    y = []
    names = []
    intervals = np.linspace(0, 20, 11)
    for i in range(len(intervals)-1):
        impacts = [0, 0]
        start = intervals[i]
        end = intervals[i+1]
        names.append(f'{start:.0f} to {end:.0f}')
        for v, rate in zip(solo_vel_list, daily_data):
            if start <= v < end:
                impacts[0] += rate
            elif start <= -v < end:
                impacts[1] += rate
        if sum(impacts) > 0:
            y.append(impacts[0]/sum(impacts))
    
    v_dust = [1/(2*Y-1)*X for X, Y in zip(x, y)]
    
    plt.bar(names, y)
    plt.plot(x,v_dust)
    plt.title('Ratio of dust detections going towards sun')
    plt.xlabel('Threshold radial velocity of Solar Orbiter [km/s]')
    plt.ylabel('Ratio of dust detections towards sun')
    plt.axhline(0.5)
    plt.show()
    
def plot_impactRate(data, packetCount):
    packetLength = 0.062
    impact_rates = []
    dates = []
    for file in data:
        for day in file:
            N_tot = packetCount[day]['packetCount']
            N_dust = len(file[day]['epoch'])
            print(N_tot)
            impact_rates.append(N_dust/(N_tot*packetLength))
            date_str = day.split('_')[3]  # Extract the date string
            date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            dates.append(date)
    
    plt.plot(dates, impact_rates, '.')
    plt.show()
            
#Set font sizes
plt.rc('font', size=28)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
# Set the font family to the LaTeX font family
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['figure.constrained_layout.use'] = True       
#Load spicepy kernels
spiceypy.furnsh('C:/Data/solo_master/kernels/lsk/naif0012.tls')
directory = "C:/Data/solo_master/kernels/spk"
for filename in os.listdir(directory):
    if filename.endswith(".bsp"):
        full_path = os.path.join(directory, filename)
        spiceypy.furnsh(full_path)

#Read data
path = 'C:\Githubs\kandidat\SOLO_orbit\SOLO_orbit_HCI.txt' #Solar orbiter distances
files = ['Labels_2020_v2.pkl', 'Labels_2021_v2.pkl', 'Labels_2022_2023_v2.pkl'] #, 'Labels_2021.pkl', 'Labels_2022.pkl', 'Labels_2023.pkl'] #files from 'label_data.py'
data = [pd.read_pickle(year) for year in files]
packetCount = pd.read_pickle('packetCounts')
#Plots
fig_path = 'C:/Users/ludwi/OneDrive - Uppsala universitet/1. Projekt/Kandidat/figures/'
plot_dust_radV(data)
