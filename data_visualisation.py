import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv('Labels.csv')
df = pd.read_pickle('Labels.pkl')

amplitudes = []
wavelenghts = []
index_amp = 3

for day in df:
    for epoch, wavelenght, waveQ in zip(df[day]['amplitude'], df[day]['wavelenght'], df[day]['waveQ']):
        amplitudes.append(max(epoch))
        if waveQ == 1:
            wavelenghts.append(max(wavelenght))

plt.hist(amplitudes)
plt.show()
plt.hist(wavelenghts)
plt.show()