import numpy as np


def weightedExpAverage(a,b):

    n = a.shape[0]

    bmax= np.max(b)

    num = 0.0
    den = 0.0
    for i in range(n):
        num += a[i]*np.exp(b[i]-bmax)
        den += np.exp(b[i]-bmax)
    
    return num/den
    
