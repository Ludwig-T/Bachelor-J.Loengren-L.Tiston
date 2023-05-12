import numpy as np

def extend_data(times, data, adaptable_noise=True):
    '''Extends data with noise'''
    np.random.seed(42)
    #Noise depending on data variance
    if adaptable_noise:
        sigma_antennas = []
        for antenna in data:
            sigma = np.std(antenna)     #Standard deviation of each antenna
            sigma_antennas.append(sigma)
            
        #Create noise for each antenna
        sigmas = np.array(sigma_antennas)
        noise1 = np.random.randn(3, 8192)*sigmas.reshape(3,1)
        noise2 = np.random.randn(3, 8192)*sigmas.reshape(3,1)
        
    else:
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
    times_doubled = times*2              

    return times_doubled, data_noise
