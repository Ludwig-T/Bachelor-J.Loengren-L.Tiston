import matplotlib.pyplot as plt
import os

from data_hadnling_L1 import get_data, get_timeseries

PATH = '//NAS24/solo/remote/data/L1'
YEAR = '2022'
MONTH = '03'   

FILE_PATH = f"{PATH}/{YEAR}/{MONTH}"

for root, dirs, files in os.walk(FILE_PATH):    #iterate folders

    for file in files:
        if 'cwf' in file:

            print(file)

            current_file_path = f"{root}/{file}"
            V, SAMPLING_RATE = get_data(current_file_path, to_extract='V')


            times= get_timeseries(SAMPLING_RATE)
            plt.plot(times, V)
            plt.show()