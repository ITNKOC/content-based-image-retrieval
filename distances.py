import numpy as np
from scipy.spatial import distance

def manhattan(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1-v2))
    return dist

def euclidean(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum(v1-v2)**2)
    return dist

def chebyshev(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1-v2))
    return dist

def canberra(v1, v2):
    return distance.canberra(v1, v2)