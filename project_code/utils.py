import numpy as np
from numpy import sin, asin, cos

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    'Haversine formula. Returns in meters. Use haversine package if higher accuracy is needed'
    r = 6378.137 * 1000 # earth radius in meter [Moritz, H. (1980). Geodetic Reference System 1980]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2-lat1)/2.0)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2.0)**2
    return 2 * r * asin(a**.5)

def resample_trajectory(x, length:int=200) -> 'np.ndarray':
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T