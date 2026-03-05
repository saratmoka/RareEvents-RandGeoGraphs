"""
This workspace is for estimating the rare events corresponds to edge count 
of the graph usning all three methods, namely, naive MC, conditional MC, and 
importance sampling 

"""

import ec
import time
#%%
IntRange = 1.0  # radius of sphere 
WindLen = 20
Level = 5
Kappa = 0.3
#MaxIter = 10**8 


#Level = int(54.0*0.1) ## For window 20*20 with intensity 0.3 and interaction range 1
#Level = int(2405*(0.5)) ### For window 20*20 with intensity 2 and interaction range 1
#Level = int(600*(0.3)) ### For window 20*20 with intensity 1 and interaction range 1
#Level = int(334*(0.2)) ### For window 15*15 with intensity 1 and interaction range 1




#%%
t0 = time.process_time()
result_nmc = ec.naiveMC(WindLen, Kappa, IntRange, Level, Seed=42)
print(f'Running time (NMC): {time.process_time() - t0:.3f}s')
print('---------------------------------------')
print('|\t \t Final results (NMC) \t \t |')
print('---------------------------------------')
print('Mean estimate Z (NMC):', ec.sci(result_nmc['mean']))
if result_nmc['mean'] != 0:
    RelVar_nmc = result_nmc['mse']/(result_nmc['mean']**2) - 1
    print('Relative variance of Y (NMC):', ec.sci(RelVar_nmc))
    print('Relative variance of Z (NMC):', ec.sci(RelVar_nmc/result_nmc['niter']))
print('Process time (sec):', round(result_nmc['time'], 3))
print('Number of iterations:', result_nmc['niter'])

#%%
t0 = time.process_time()
result_cmc = ec.conditionalMC(WindLen, Kappa, IntRange, Level, Seed=42)
print(f'Running time (CMC): {time.process_time() - t0:.3f}s')
RelVar_cmc = result_cmc['mse']/(result_cmc['mean']**2) - 1
print('--------------------------------------')
print('|\t \t Final results (CMC) \t \t|')
print('--------------------------------------')
print('Mean estimate Z (CMC):', ec.sci(result_cmc['mean']))
print('Relative variance of Y (CMC):', ec.sci(RelVar_cmc))
print('Relative variance of Z (CMC):', ec.sci(RelVar_cmc/result_cmc['niter']))
print('Process time (sec):', round(result_cmc['time'], 3))
print('Number of iterations:', result_cmc['niter'])

#%%
GridRes = 20 # the number of grid cells per interaction range

t0 = time.process_time()
result_ismc = ec.ISMC(WindLen, GridRes, Kappa, IntRange, Level, Seed=42)
print(f'Running time (IS): {time.process_time() - t0:.3f}s')
RelVar_ismc = result_ismc['mse']/(result_ismc['mean']**2) - 1
print('-----------------------------------')
print('|\t \t Final results (IS) \t \t |')
print('-----------------------------------')
print('Mean estimate Z (IS):', ec.sci(result_ismc['mean']))
print('Relative variance of Y (IS):', ec.sci(RelVar_ismc))
print('Relative variance of Z (IS):', ec.sci(RelVar_ismc/result_ismc['niter']))
print('Process time (sec):', round(result_ismc['time'], 3))
print('Number of iterations:', result_ismc['niter'])


