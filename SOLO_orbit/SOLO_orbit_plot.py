import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

filename = 'SOLO_orbit_HCI.txt'
path = Path(__file__).with_name(filename) 


df = pd.read_csv(path, sep=' ')
# combine the year, month, and day columns into a single datetime column
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])

# group the data by date and calculate the mean of R[AU] for each date
daily_mean = df.groupby('datetime')['R[AU]'].mean()

plt.plot(daily_mean.index, daily_mean.values)
plt.title('Mean R[AU] per day')
plt.xlabel('Date')
plt.ylabel('Mean R[AU]')
plt.show()