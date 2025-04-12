#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July 3 2023

@author: Dr Sarat Moka, The School of Mathematics and Statistics, University of New South Wales.

s.moka@unsw.edu.au, https://saratmoka.com

This code provides Naive MC, Conditional MC, and Importance Sampling MC for estimation 
the edge count related rare-events in random geometric graphs, also known as Gilbert Graphs. 
Refer to the paper Moka, Christian, Schimdt, and Kroese (2025) for more details.

"""
#import sys 
#import os
#import datetime
#import bisect
import time
#from tqdm import tqdm
#import matplotlib.pyplot as plt
#from matplotlib import colors
#import matplotlib.patches as patches
#import pandas as pd
import numpy as np
from numpy import linalg
from scipy.stats import poisson
from IPython.display import clear_output

sqrt_2 = np.sqrt(2) ## Compute square root of 2 just once.

def sci(x, digits=2):
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")

'''
Counts edges between new point and existing points
'''
def newEdges_bin_version(NewPoint, BinnedPoints, IntRange, BinEdg, nBins):

    count = 0
    bin_x = int(NewPoint[0]/BinEdg)
    bin_y = int(NewPoint[1]/BinEdg)
    
    for i in range(max(0, bin_x-1), min(bin_x+2, nBins)):
        for j in range(max(0, bin_y-1), min(bin_y+2, nBins)):
            for pt in BinnedPoints[i][j]:
                if linalg.norm(pt - NewPoint) < IntRange:
                    count += 1
    return count

'''
Naive MC
'''
def naiveMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp = 100000, Tol=0.001):
    """
    Implementation of the naive Monte Carlo simulation method for estimating the rare-event 
    probabilities on the edge count. 

    Parameters
    ----------
    WindLen : float
        Length of each side of the square window (lambda).
    Kappa : float
        Intensity of the the Poisson point process.
    IntRange : float
        Interation range create edges in the Geometric random graph. 
    Level : int 
        Threshold parameter ell in the rare-event definition on edge count.
    MaxIter : int
        Maximum number iterations. Algorithm terminates irrespective convergence once 
        the number of iterations reaches this number. 
        Default value 10**8
    WarmUp : int
        Minimum number of iterations used in the algorithm. Termination conditions are 
        checked only after these many iterations. 
        Default value 100000.
    Tol : float
        Tolerance used for termination. Algorithm terminates when the relative variance of 
        the mean estimator Z falls below Tol consecutively over 100 iterations. 
        Default value 0.001.

    Returns
    -------
    result : dictionary
    
             result["mean"] : Sample mean estimate
             result["mse"] : Sample mean square estimate
             result["time"] : Process time taken by the algorithm
             result["niter"] : Number of iterations before termination

    """

    ExpPoiCount = Kappa*(WindLen**2) ## Expected number of Poisson points.
    BinSize = int(WindLen/IntRange)     ## Number of bins. 
    BinEdg = WindLen/BinSize            ## Side length of each bin.
    MeanEst = 0.0                          ## Empirical average of the estimator.
    Time = 0.0
    
    Patience = 0
    l = 0
    stop = False
    print("Warming up ...... ")
    
    while not stop:
        tic = time.process_time()
        l += 1
        BinnedPoints = [[[] for _ in range(BinSize)] for _ in range(BinSize)]
        EdgeCount = 0
        N = np.random.poisson(ExpPoiCount)
        Y = 1
        n = 1
        while Y == 1 and n <= N:
            NewPoint = WindLen*np.random.random_sample(2)
            n = n + 1
            EdgeCount = EdgeCount + newEdges_bin_version(NewPoint, BinnedPoints, IntRange, BinEdg, BinSize)
            bin_x = int(NewPoint[0]/BinEdg)
            bin_y = int(NewPoint[1]/BinEdg)
            BinnedPoints[bin_x][bin_y].append(NewPoint)
            if EdgeCount > Level:
                Y = 0
                
        MeanEst = ((l-1)*MeanEst + Y)/l
        
        if MeanEst != 0:
            RV = 1/MeanEst - 1
        else:
            RV = np.inf

        toc = time.process_time()
        Time += toc - tic
        
        if l%WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration: ', l, '-----')
            print('\nMean estimate Z (NMC):', sci(MeanEst)) 
            print('Relative variance of Y (NMC):', sci(RV)) 
            print('Relative variance of Z (NMC):', sci(RV/l)) 
                
        if l >= WarmUp:
            if RV/l < Tol:
                Patience += 1
            else:
                Patience = 0 
        # else:
        #     num_dots = int(l // (WarmUp/20))   # Goes from 0 to 10
        #     dots = '=' * num_dots+'>'
        #     message = f'\rWarm up in progress |{dots:<20}|'  # pad to keep length constant
        #     sys.stdout.write(message)
        #     sys.stdout.flush()            
            
            
        if Patience >= 100 or l >= MaxIter:
            stop = True
    
    # Collect the results
    result = {
		"mean" : MeanEst,
		"mse" : MeanEst,
		"time" : Time,
		"niter" : l,
		}
    
    return result


'''
Conditional MC
'''
def conditionalMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp = 10000, Tol=0.001):
    """
    Implementation of the conditional Monte Carlo simulation method for estimating the rare-event 
    probabilities on the edge count. 

    Parameters
    ----------
    WindLen : float
        Length of each side of the square window (lambda).
    Kappa : float
        Intensity of the the Poisson point process.
    IntRange : float
        Interation range create edges in the Geometric random graph. 
    Level : int 
        Threshold parameter ell in the rare-event definition on edge count.
    MaxIter : int
        Maximum number iterations. Algorithm terminates irrespective convergence once 
        the number of iterations reaches this number. 
        Default value 10**8
    WarmUp : int
        Minimum number of iterations used in the algorithm. Termination conditions are 
        checked only after these many iterations. 
        Default value 10000.
    Tol : float
        Tolerance used for termination. Algorithm terminates when the relative variance of 
        the mean estimator Z falls below Tol consecutively over 100 iterations. 
        Default value 0.001.

    Returns
    -------
    result : dictionary
    
             result["mean"] : Sample mean estimate
             result["mse"] : Sample mean square estimate
             result["time"] : Process time taken by the algorithm
             result["niter"] : Number of iterations before termination

    """

    ExpPoissonCount = Kappa*(WindLen**2) ## Expected number of Poisson points.
    nBins = int(WindLen/IntRange)     ## Number of bins.
    BinEdg = WindLen/nBins            ## Side length of each bin.
    MeanEst = 0.0                     ## Sample mean of the estimator
    MeanSqrEst = 0.0                  ## Sample mean of the suqares of estimator
    Time = 0.0
    
    Patience = 0
    l = 0
    stop = False
    print("Warming up ...... ")
    
    while not stop:
        tic = time.process_time()
        l += 1        
        BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
        EdgeCount = 0
        n = 0
        while EdgeCount <= Level:
            NewPoint = WindLen*np.random.random_sample(2)
            n = n + 1
            EdgeCount = EdgeCount + newEdges_bin_version(NewPoint, BinnedPoints, IntRange, BinEdg, nBins)
            bin_x = int(NewPoint[0]/BinEdg)
            bin_y = int(NewPoint[1]/BinEdg)
            BinnedPoints[bin_x][bin_y].append(NewPoint)
                
        Y_hat = poisson.cdf(n - 1, ExpPoissonCount)
        MeanEst = ((l-1)*MeanEst + Y_hat)/l
        MeanSqrEst = ((l-1)*MeanSqrEst + Y_hat*Y_hat)/l
        RV = MeanSqrEst/(MeanEst**2) - 1
        toc = time.process_time()
        Time += toc - tic
        
        if l%WarmUp == 0:
            
            clear_output(wait=True)
            print('\n----- Iteration: ', l, '-----')
            print('\nMean estimate Z (CMC):', sci(MeanEst))
            print('Relative variance of Y_hat (CMC):', sci(RV)) 
            print('Relative variance of Z (CMC):', sci(RV/l)) 
            
        if l >= WarmUp:
            if RV/l < Tol:
                Patience += 1
            else:
                Patience = 0  
            
        if Patience >= 100 or l >= MaxIter:
            stop = True
        
    
    # Collect the results
    result = {
		"mean" : MeanEst,
		"mse" : MeanSqrEst,
		"time" : Time,
		"niter" : l,
		}
    return result



''' 
Distance between given two cells
'''
def distBtwCells(xx, yy):
    dist = np.linalg.norm(xx - yy)
    dist = max(dist, np.linalg.norm(xx + [0,1] - yy))
    dist = max(dist, np.linalg.norm(xx + [1,1] - yy))
    dist = max(dist, np.linalg.norm(xx + [1,0] - yy))
    
    dist = max(dist, np.linalg.norm(xx - yy - [0,1]))
    dist = max(dist, np.linalg.norm(xx - yy - [1,1]))
    dist = max(dist, np.linalg.norm(xx - yy - [1,0]))
    
    return dist

'''
Identifying the neighbors of each event in the dependency graph
'''
def generateNeighborsMatrix(GridEdg, IntRange):
    spread = int(IntRange/GridEdg) - 1
    if spread < 0:
        print("Error: cell digonal length is bigger than IntRange")
        
    arr_size = 2*spread + 1
    Neighbors = np.zeros((arr_size, arr_size), dtype=int)
    center = np.array((spread, spread))
    for x in range(arr_size):
        for y in range(arr_size):
            dist = distBtwCells(np.array((x,y)), center)
            if dist*GridEdg <= IntRange:
                Neighbors[x][y] = 1 ## the cell is completely inside
    return Neighbors

'''
Generates the next point.
'''
def generateNextPoint(GridSize, GridEdg, IntRange, EdgeCount, OrderMatrix, Neighbors, Level):
    
    tic0 = time.time()
    NonBlockCells = np.argwhere(OrderMatrix <= Level - EdgeCount)
    toc0 = time.time()
    NonBlockCount = NonBlockCells.shape[0]
    if NonBlockCount > 0:
        Index = np.random.randint(NonBlockCells.shape[0])
        X_ind = NonBlockCells[Index][0]
        Y_ind = NonBlockCells[Index][1]
    
    
        
        NewPoint = (np.array([X_ind, Y_ind]) + np.random.random_sample(2))*GridEdg
        spread = int((Neighbors.shape[0] - 1)/2)
        
        x_left = min(X_ind, spread)
        x_right = min(GridSize - X_ind, spread + 1)
        y_left = min(Y_ind, spread)
        y_right = min(GridSize - Y_ind, spread + 1)
            
        OrderMatrix[X_ind - x_left:X_ind + x_right, Y_ind - y_left: Y_ind + y_right] += Neighbors[spread - x_left:spread + x_right, spread - y_left: spread + y_right]
    else:
        NewPoint = None
        
    return NewPoint, NonBlockCount, toc0 - tic0

''' 
Counts edges between new point and existing points
'''
def newEdges(NewPoint, BinnedPoints, IntRange, BinEdg, nBins):

    count = 0
    bin_x = int(NewPoint[0]/BinEdg)
    bin_y = int(NewPoint[1]/BinEdg)
    
    for i in range(max(0, bin_x-1), min(bin_x+2, nBins)):
        for j in range(max(0, bin_y-1), min(bin_y+2, nBins)):
            for pt in BinnedPoints[i][j]:
                if linalg.norm(pt - NewPoint) < IntRange:
                    count += 1
    return count 

'''
 IS implementation 
'''
def ISMC(WindLen, GridRes, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=1000, Tol=0.001):
    """
    Implementation of the importance sampling based Monte Carlo simulation method for estimating the rare-event 
    probabilities on the edge count. 

    Parameters
    ----------
    WindLen : float
        Length of each side of the square window (lambda).
    GridRes : int
        Resolution of the grid, i.e. the number of grid cells per unit length.
    Kappa : float
        Intensity of the the Poisson point process.
    IntRange : float
        Interation range create edges in the Geometric random graph. 
    Level : int 
        Threshold parameter ell in the rare-event definition on edge count.
    MaxIter : int
        Maximum number iterations. Algorithm terminates irrespective convergence once 
        the number of iterations reaches this number. 
        Default value 10**8
    WarmUp : int
        Minimum number of iterations used in the algorithm. Termination conditions are 
        checked only after these many iterations. 
        Default value 1000.
    Tol : float
        Tolerance used for termination. Algorithm terminates when the relative variance of 
        the mean estimator Z falls below Tol consecutively over 100 iterations. 
        Default value 0.001.

    Returns
    -------
    result : dictionary
    
             result["mean"] : Sample mean estimate
             result["mse"] : Sample mean square estimate
             result["time"] : Process time taken by the algorithm
             result["niter"] : Number of iterations before termination

    """
    
    ExpPoissonCount = Kappa*(WindLen**2) ## Expected number of Poisson points.
    nBins = int(WindLen/IntRange)     ## Number of bins.
    GridSize = int(nBins*GridRes)      ## Grid size on the whole window.
    GridEdg = WindLen/GridSize
    BinEdg = WindLen/nBins
    npts = int(2*ExpPoissonCount)
    q = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts+1)])
    
    MeanEst = 0.0                     ## Sample mean of the estimator
    MeanSqrEst = 0.0                  ## Sample mean of the suqares of estimator 
    Time = 0.0
    
    Neighbors = generateNeighborsMatrix(GridEdg, IntRange)  
    
    Patience = 0
    l = 0
    stop = False
    print("Warming up ...... ")
    
    while not stop:
        tic = time.process_time()                 ## Process time start
        l += 1
        BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
        OrderMatrix = np.zeros((GridSize, GridSize), dtype=int)
        EdgeCount = 0
        NonBlockCount = int(GridSize**2)
        LHR = np.zeros(npts+1)
        LHR[0] = 1.0
        n = 0
        while EdgeCount <= Level and NonBlockCount > 0 and n < npts:
            
            NewPoint, NonBlockCount, time_gen = generateNextPoint(GridSize, GridEdg, IntRange, EdgeCount, OrderMatrix, Neighbors, Level)
            n += 1
            LHR[n] = LHR[n-1]*(NonBlockCount/(GridSize**2)) 
            if NonBlockCount > 0:
                EdgeCount += newEdges(NewPoint, BinnedPoints, IntRange, BinEdg, nBins)       
                if EdgeCount > Level:
                    LHR[n] = 0.0
                    
                bin_x = int(NewPoint[0]/BinEdg)
                bin_y = int(NewPoint[1]/BinEdg)
                BinnedPoints[bin_x][bin_y].append(NewPoint)
                
        Y_tilde = q@LHR
        MeanEst = (l*MeanEst + Y_tilde)/(l+1)
        MeanSqrEst = (l*MeanSqrEst + Y_tilde*Y_tilde)/(l+1)
        RV = MeanSqrEst/(MeanEst**2) - 1
        toc = time.process_time()  # Process time ends
        Time += toc - tic
        
        if l%WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration: ', l, '-----')
            print('\nGrid size:', GridSize,'x', GridSize)
            print('Mean estimate Z (IS):', sci(MeanEst)) 
            print('Relative variance of Y_tilde (IS):', sci(RV))
            print('Relative variance of Z (IS):', sci(RV/l))
            
            
        if l >= WarmUp:
            if RV/l < Tol:
                Patience += 1
            else:
                Patience = 0  
            
        if Patience >= 100 or l >= MaxIter:
            stop = True
            

    
    # Collect the results
    result = {
		"mean" : MeanEst,
		"mse" : MeanSqrEst,
		"time" : Time,
		"niter" : l,
		}
    
    return result
