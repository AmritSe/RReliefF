import relieff
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as sio

# Try a regression problem (RReliefF) or a classification problem (Relief and ReliefF)
regressionProblem = False

if regressionProblem:
    x = np.linspace(0, 5, 50)
    y = x

    xx, yy = np.meshgrid(x, y)
    mm = np.random.rand(xx.shape[0], xx.shape[1])
    zz = 5 * xx**2 + 5 * yy**2

    X = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1), mm.reshape(-1,1)], 1)
    y = zz.reshape(-1, 1)

    W = relieff.RReliefF(X, y)
    print(W)

else:
    x = np.linspace(0, 5, 50)
    y = (x**2 + 2)>10


    z = 5 * np.random.rand(x.shape[0])
    X = np.concatenate([x.reshape(-1,1), z.reshape(-1,1)],1)

    WRelief = relieff.Relief(X, y)
    WReliefF = relieff.ReliefF(X, y)
    
    print(WRelief)
    print(WReliefF)
