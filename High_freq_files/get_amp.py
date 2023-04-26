import matplotlib.pyplot as plt
from data_handling import tripple_plot

def get_amp(X, Y, to_plot=False):
    '''Returns the minimum and maximum amplitud for each antenna, and plots them if needed.
    Input is timedata X, and a list of the three electric field Y.'''
    minAmps = [min(Y[0]), min(Y[1]), min(Y[2])]
    maxAmps = [max(Y[0]), max(Y[1]), max(Y[2])]
    if to_plot:
        fig, axes = tripple_plot(X, Y, to_show=False)
        for i, ax in enumerate(axes):
            ax.axhline(minAmps[i], color='r', ls='--')
            ax.axhline(maxAmps[i], color='r', ls='--')
        plt.show()
    return list(zip(minAmps, maxAmps))
