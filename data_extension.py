import numpy as np

def extend_data(times, data):
    np.random.seed(42)
    sigma = 0.0005 #parameter that can be tweaked, 

    #generate noise
    noise1 = np.random.randn(3, 8192)*sigma
    noise2 = np.random.randn(3, 8192)*sigma 

    #add noise to data
    data_noise1 = np.hstack((noise1, data))      
    data_noice_full = np.hstack((data_noise1, noise2))

    #extract every other element to fit model
    data_noise= [new_data[::2] for new_data in data_noice_full]
    
    #extend time to fit extended data
    times_doubbled = times*2              

    return times_doubbled, data_noise
