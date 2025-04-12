#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 03:21:10 2024

@author: Dr Sarat Moka, The School of Mathematics and Statistics, University of New South Wales.

s.moka@unsw.edu.au, https://saratmoka.com

This code provides Naive MC, Conditional MC, and Importance Sampling MC for estimation 
the max degree related rare-events in random geometric graphs, also known as Gilbert Graphs. 
Refer to the paper Moka, et al. [2024] for more details.

"""
#import sys
import networkx as nx
import numpy as np
#from numpy import linalg
from scipy.stats import poisson
from scipy.spatial import distance_matrix
import time
#from tqdm import tqdm
from IPython.display import clear_output

def sci(x, digits=2):
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")

#%%
'''
Counts degree of each point in the graph
'''

def maxDegree(NumPoints, WindLen, IntRange):
    """
    Calculate the maximum degree of a random geometric graph in a square window.

    Parameters:
    - NumPoints: Number of points in the graph.
    - WindLen: Size of the square window (i.e., points are placed in [0, window_size] x [0, window_size]).
    - IntRange: Distance threshold to determine edges between points.

    Returns:
    - MaxDeg: The maximum degree of any node in the graph.
    """

    # Generate random points within the square window
    points = np.random.uniform(0, WindLen, (NumPoints, 2))
    
    # Compute pairwise distances between all points
    distances = distance_matrix(points, points)
    
    # Create adjacency matrix based on distance threshold
    adjacency_matrix = (distances < IntRange) & (distances > 0)
    
    # Calculate degree of each node
    degrees = adjacency_matrix.sum(axis=1)
    
    # Return the maximum degree
    MaxDeg = np.max(degrees)
    
    return MaxDeg


#%%
'''
Naive MC
'''
def naiveMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp = 100000, Tol=0.001):
    """
    Implementation of the naive Monte Carlo simulation method for estimating the rare-event 
    probabilities on the maximum degree. 

    Parameters
    ----------
    WindLen : float
        Length of each side of the square window (lambda).
    Kappa : float
        Intensity of the the Poisson point process.
    IntRange : float
        Interation range create edges in the Geometric random graph. 
    Level : int 
        Threshold parameter ell in the rare-event definition on maximum degree.
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
    
             result["mean"]: Sample mean estimate
             result["mse"]: Sample mean square estimate
             result["time"]: Process time taken by the algorithm
             result["niter"]: Number of iterations before termination

    """
    
    ExpPoiCount = Kappa*(WindLen**2) ## Expected number of Poisson points.
    MeanEst = 0.0                          ## Empirical average of the estimator.
    Time = 0.0
    
    Patience = 0
    l = 0
    stop = False
    print("Warming up ...... ")
    
    while not stop:
        tic = time.process_time()
        l += 1
        N = np.random.poisson(ExpPoiCount)
        md = maxDegree(N, WindLen, IntRange)       
        if md > Level:
            Y = 0
        else:
            Y = 1
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


#%%
def generatePointsUntilMaxDegree(MaxNumPoints, WindLen, IntRange, Level):
    """
    Generate points sequentially in a random geometric graph until 
    the maximum degree exceeds a given level.

    Parameters
    ----------
    
    MaxNumPoints : int
        Maximum number of points to consider.
    WindLen : float
        Length of each side of the square window (lambda).
    IntRange : float
        Interation range create edges in the Geometric random graph.  
    Level : int 
        Threshold parameter ell in the rare-event definition on maximum degree.

    Returns
    -------
    MaxDeg : int
        Maximum degree reached.
    n : 
        Number of points generated until the max degree exceeded the level or MaxNumPoints.
    """

    points = []  # Store points sequentially

    for i in range(MaxNumPoints):
        # Generate a new random point within the window
        new_point = np.random.uniform(0, WindLen, (1, 2))
        points.append(new_point[0])

        # Convert points to an array for distance calculations
        points_array = np.array(points)
        
        # Compute the distance of the new point to all previous points
        distances = distance_matrix(points_array, points_array)

        # Create adjacency matrix based on distance threshold
        adjacency_matrix = (distances < IntRange) & (distances > 0)
        
        # Calculate the degree of each node
        degrees = adjacency_matrix.sum(axis=1)
        
        # Update the maximum degree
        MaxDeg = np.max(degrees)

        # Check if the max degree exceeds the given level
        if MaxDeg > Level:
            return MaxDeg, i 

    # If we reach the limit without exceeding level, return all points
    return MaxDeg, MaxNumPoints

'''
Conditional MC
'''
def conditionalMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp = 1000, Tol=0.001):
    """
    Implementation of the conditional Monte Carlo simulation method for estimating the rare-event 
    probabilities on the maximum degree. 

    Parameters
    ----------
    WindLen : float
        Length of each side of the square window (lambda).
    Kappa : float
        Intensity of the the Poisson point process.
    IntRange : float
        Interation range create edges in the Geometric random graph. 
    Level : int 
        Threshold parameter ell in the rare-event definition on maximum degree.
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
    
             result["mean"]: Sample mean estimate
             result["mse"]: Sample mean square estimate
             result["time"]: Process time taken by the algorithm
             result["niter"]: Number of iterations before termination

    """

    ExpPoissonCount = Kappa*(WindLen**2) ## Expected number of Poisson points.
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
        MaxDeg, n = generatePointsUntilMaxDegree(int(3*ExpPoissonCount), WindLen, IntRange, Level)
                
        Y_hat = poisson.cdf(n, ExpPoissonCount)
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


#%%

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
def generateNeighbors(GridEdg, IntRange):
    spread = int(IntRange/GridEdg) - 1
    if spread < 0:
        print("Error: cell digonal length is bigger than IntRange")
        
    arr_size = 2*spread + 1
    Neighbors = []
    center = np.array((spread, spread))
    for x in range(arr_size):
        for y in range(arr_size):
            cell = np.array((x,y))
            dist = distBtwCells(cell, center)
            if dist*GridEdg <= IntRange:
                Neighbors.append(cell-center) ## the cell is completely inside
    return Neighbors

#%%
"""
Update the graph by adding the new point and also update the block matrix
"""
def update(G, Point, Index, IntRange, Neighbors, BlockMatrix, OrderMatrix, Level):
    """
    Adds a node to graph G at the given 2D point and connects it to existing nodes
    within the given Euclidean distance InteRange.

    Parameters:
    - G: networkx.Graph
        The existing graph.
    - Point: tuple (x, y)
        The coordinates of the new node.
    - InteRange: float
        The maximum distance for creating an edge.

    Returns:
    - None (modifies G in place)
    """
    GridSize = BlockMatrix.shape[0]
    node_id = len(G.nodes)  # Assign a unique ID to the new node
    G.add_node(node_id, pt=Point, ind = Index, stat = False)
    
        
    # Check distance to existing nodes
    for i, data in G.nodes(data=True):
        if i == node_id:
            continue  # Skip self

        other_point = data["pt"]
        distance = np.linalg.norm(np.array(Point) - np.array(other_point))

        if distance <= IntRange:
            G.add_edge(node_id, i)
    
    #Update the order matrix
    for nb in Neighbors:
        ind = nb + Index
        if 0 <= ind[0] < GridSize and 0 <= ind[1] < GridSize:
            OrderMatrix[ind[0], ind[1]] += 1
            # if OrderMatrix[ind[0], ind[1]] > Level:
            #     BlockMatrix[ind[0], ind[1]] = 1
    BlockMatrix[OrderMatrix > Level] = 1
    
    # Update the blocking matrix
    stop = False
    for i, data in G.nodes(data=True):
        deg = G.degree(i)
        if deg >= Level and data["stat"] == False:
            data["stat"] = True
            for nb in Neighbors:
                ind = data["ind"] + nb
                if 0 <= ind[0] < GridSize and 0 <= ind[1] < GridSize:
                    BlockMatrix[ind[0], ind[1]] = 1
        if deg > Level:
            stop = True
        
    return stop
    

"""
Importance Sampling Monte Carlo estimation
"""            
def ISMC(WindLen, GridRes, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=100, Tol=0.001):
    """
    Implementation of the importance sampling based Monte Carlo simulation method for estimating the rare-event 
    probabilities on the maximum degree. 

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
        Threshold parameter ell in the rare-event definition on maximum degree.
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
    
             result["mean"]: Sample mean estimate
             result["mse"]: Sample mean square estimate
             result["time"]: Process time taken by the algorithm
             result["niter"]: Number of iterations before termination

    """
    
    ExpPoissonCount = Kappa*(WindLen**2) ## Expected number of Poisson points.
    GridSize = int(int(WindLen/IntRange)*GridRes)      ## Grid size on the window.
    GridEdg = WindLen/GridSize
    npts = int(3*ExpPoissonCount)
    q = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts+1)])
    
    MeanEst = 0.0                     ## Sample mean of the estimator
    MeanSqrEst = 0.0                  ## Sample mean of the suqares of estimator 
    Time = 0.0
    Neighbors = generateNeighbors(GridEdg, IntRange)

    Patience = 0
    l = 0
    stop = False
    print("Warming up ...... ")
    
    while not stop:
        tic = time.process_time() 
        l += 1
        LHR = np.zeros(npts+1)
        LHR[0] = 1.0
        BlockMatrix = np.zeros((GridSize, GridSize), dtype=int) ## (i, j) element of this matrix is zero if that cell is non-blocked, otherwise zero.
        OrderMatrix = np.zeros((GridSize, GridSize), dtype=int)
        
        G = nx.Graph()  # Start with an empty graph
        stop_in_loop = False
        
        for n in range(npts):
            # Find indices where the matrix has 1s
            NonBlockIndices = list(np.argwhere(BlockMatrix == 0))
            
            # Select one index uniformly at random
            NonBlockCount = len(NonBlockIndices)
            if NonBlockCount > 0:
                # Get the cell number for genertaing a new point
                Index = NonBlockIndices[np.random.choice(NonBlockCount)]
                
                # Generate the next point
                Point = (Index + np.random.random_sample(2))*GridEdg

                # update the graph and the block matrix
                stop_in_loop = update(G, Point, Index, IntRange, Neighbors, BlockMatrix, OrderMatrix, Level)
                
                if not stop_in_loop:
                    LHR[n+1] = LHR[n]*(NonBlockCount/(GridSize**2))
                    
            else:
                stop_in_loop = True
                
            if stop_in_loop:
                break
            
        Y_tilde = q@LHR
        MeanEst = ((l-1)*MeanEst + Y_tilde)/l
        MeanSqrEst = ((l-1)*MeanSqrEst + Y_tilde*Y_tilde)/l
        RV = MeanSqrEst/(MeanEst**2) - 1
        toc = time.process_time() 
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

