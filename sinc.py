import numpy as np

def sinc_interp(interpolate, num_instances, unsampled):
    # Find the period
    T = num_instances[1] - num_instances[0]

    sincM = np.tile(unsampled, (len(num_instances), 1)) - np.tile(num_instances[:, np.newaxis], (1, len(unsampled)))
    y = np.dot(x, np.sinc(sincM/T))
    return y
