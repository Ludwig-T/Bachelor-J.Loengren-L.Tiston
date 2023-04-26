import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('Labels_2022.pkl')

amplitudes = []
wavelenghts = []

for day in df:
    for amplitude, wavelenght, waveQ in zip(df[day]['amplitude'], df[day]['wavelenght'], df[day]['waveQ']):
        amplitudes.append(max(amplitude))
        good_wavel = np.array(wavelenght)[np.array(waveQ, dtype=bool)]
        if len(good_wavel) != 0:
            wavelenghts.append(max(good_wavel))

plt.hist(amplitudes)
plt.show()
plt.hist(wavelenghts)
plt.show()