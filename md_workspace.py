"""
This workspace is for estimating the rare events corresponds to maximum degree 
of the graph usning all three methods, namely, naive MC, conditional MC, and 
importance sampling 

"""
import md
import time
#%%
IntRange = 1.0  # radius of sphere 
WindLen = 10
Level = 0
Kappa = 0.4
#MaxIter = 10**8


#%%

t0 = time.process_time()
result_nmc = md.naiveMC(WindLen, Kappa, IntRange, Level, Seed=42)
print(f'Running time (NMC): {time.process_time() - t0:.3f}s')
print('---------------------------------------')
print('|\t \t Final results (NMC) \t \t |')
print('---------------------------------------')
print('Mean estimate Z (NMC):', md.sci(result_nmc['mean']))
if result_nmc['mean'] != 0:
    RelVar_nmc = result_nmc['mse']/(result_nmc['mean']**2) - 1
    print('Relative variance of Y (NMC):', md.sci(RelVar_nmc))
    print('Relative variance of Z (NMC):', md.sci(RelVar_nmc/result_nmc['niter']))
print('Process time (sec):', round(result_nmc['time'], 3))
print('Number of iterations:', result_nmc['niter'])

#%%
t0 = time.process_time()
result_cmc = md.conditionalMC(WindLen, Kappa, IntRange, Level, Seed=42)
print(f'Running time (CMC): {time.process_time() - t0:.3f}s')
RelVar_cmc = result_cmc['mse']/(result_cmc['mean']**2) - 1
print('--------------------------------------')
print('|\t \t Final results (CMC) \t \t|')
print('--------------------------------------')
print('Mean estimate Z (CMC):', md.sci(result_cmc['mean']))
print('Relative variance of Y (CMC):', md.sci(RelVar_cmc))
print('Relative variance of Z (CMC):', md.sci(RelVar_cmc/result_cmc['niter']))
print('Process time (sec):', round(result_cmc['time'], 3))
print('Number of iterations:', result_cmc['niter'])

#%%
GridRes = 10 # the number of grid cells per interaction range

t0 = time.process_time()
result_ismc = md.ISMC(WindLen, GridRes, Kappa, IntRange, Level, Seed=42)
print(f'Running time (IS): {time.process_time() - t0:.3f}s')
RelVar_ismc = result_ismc['mse']/(result_ismc['mean']**2) - 1
print('-----------------------------------')
print('|\t \t Final results (IS) \t \t |')
print('-----------------------------------')
print('Mean estimate Z (IS):', md.sci(result_ismc['mean']))
print('Relative variance of Y (IS):', md.sci(RelVar_ismc))
print('Relative variance of Z (IS):', md.sci(RelVar_ismc/result_ismc['niter']))
print('Process time (sec):', round(result_ismc['time'], 3))
print('Number of iterations:', result_ismc['niter'])



