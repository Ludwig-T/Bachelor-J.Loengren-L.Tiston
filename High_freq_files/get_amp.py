import matplotlib.pyplot as plt
from data_handling import tripple_plot
import numpy as np
import scipy.signal as scisig
def get_amp(X, Y, to_plot=False):
    '''Returns the minimum and maximum amplitud for each antenna, and plots them if needed.
    Input is timedata X, and a list of the three electric field Y.'''
    F = 21
    medAmps = [np.median(scisig.medfilt(y, kernel_size=F)) for y in Y]
    minAmps = [min(Y[0]), min(Y[1]), min(Y[2])]
    maxAmps = [max(Y[0]), max(Y[1]), max(Y[2])]
    
    a_max = 0
    for y in Y:
        mx = np.max(np.abs(y))
        if mx > a_max:
            a_max = mx
            t_max = X[np.argmax(np.abs(y))]
    t_left = t_max - 2
    t_right = t_max + 5
    mask = np.logical_and(X >= t_left, X <= t_right)
    indices = np.where(mask)[0]
    
    Ts_min = []
    Ts_max = []
    for y in Y:
        min_index = indices[np.argmin(y[indices])]
        Ts_min.append(X[min_index])
        max_index = indices[np.argmax(y[indices])]
        Ts_max.append(X[max_index])

    if to_plot:
        fig, axes = tripple_plot(X, Y, to_show=False)
        for i, ax in enumerate(axes):
            #ax.axhline(minAmps[i], color='r', ls='--')
            #ax.axhline(maxAmps[i], color='r', ls='--')
            ax.axvline(Ts_min[i], color='g', ls='--')
            ax.axvline(Ts_max[i], color='g', ls='-')
        plt.show()
    return list(zip(minAmps, maxAmps, medAmps, Ts_min, Ts_max))
