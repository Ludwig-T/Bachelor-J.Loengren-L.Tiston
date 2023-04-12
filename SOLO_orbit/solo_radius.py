import pandas as pd
from pathlib import Path

filename = 'SOLO_orbit_HCI.txt'
path = Path(__file__).with_name(filename) 


def get_radius(date_str):
    """
    Takes a date in format 'YYYY-MM-DD'
    and returns the averaged radius of SOLO to the sun
    in Astronomical units
    """
    df = pd.read_csv(path, sep=' ')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
    daily_mean = df.groupby('datetime')['R[AU]'].mean()

    return daily_mean[date_str]

y = '2020-05-05'
print(get_radius(y))
