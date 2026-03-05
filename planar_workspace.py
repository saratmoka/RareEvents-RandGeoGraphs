"""
This workspace is for estimating the rare event {Gilbert graph is planar}
using all three methods: naive MC, conditional MC, and importance sampling.
"""

import planar
import time
#%%
IntRange = 1.0  # interaction range
WindLen = 10
Kappa = 0.3
GridRes = 10   # 100x100 grid

#%%
t0 = time.process_time()
result_nmc = planar.naiveMC(WindLen, Kappa, IntRange, Seed=42)
print(f'Running time (NMC): {time.process_time() - t0:.3f}s')
print('---------------------------------------')
print('|\t \t Final results (NMC) \t \t |')
print('---------------------------------------')
print('Mean estimate Z (NMC):', planar.sci(result_nmc['mean']))
if result_nmc['mean'] != 0:
    RelVar_nmc = result_nmc['mse']/(result_nmc['mean']**2) - 1
    print('Relative variance of Y (NMC):', planar.sci(RelVar_nmc))
    print('Relative variance of Z (NMC):', planar.sci(RelVar_nmc/result_nmc['niter']))
print('Process time (sec):', round(result_nmc['time'], 3))
print('Number of iterations:', result_nmc['niter'])

#%%
t0 = time.process_time()
result_cmc = planar.conditionalMC(WindLen, Kappa, IntRange, Seed=42)
print(f'Running time (CMC): {time.process_time() - t0:.3f}s')
RelVar_cmc = result_cmc['mse']/(result_cmc['mean']**2) - 1
print('--------------------------------------')
print('|\t \t Final results (CMC) \t \t|')
print('--------------------------------------')
print('Mean estimate Z (CMC):', planar.sci(result_cmc['mean']))
print('Relative variance of Y (CMC):', planar.sci(RelVar_cmc))
print('Relative variance of Z (CMC):', planar.sci(RelVar_cmc/result_cmc['niter']))
print('Process time (sec):', round(result_cmc['time'], 3))
print('Number of iterations:', result_cmc['niter'])

#%%
t0 = time.process_time()
result_ismc = planar.ISMC(WindLen, GridRes, Kappa, IntRange, Seed=42)
print(f'Running time (IS): {time.process_time() - t0:.3f}s')
RelVar_ismc = result_ismc['mse']/(result_ismc['mean']**2) - 1
print('-----------------------------------')
print('|\t \t Final results (IS) \t \t |')
print('-----------------------------------')
print('Mean estimate Z (IS):', planar.sci(result_ismc['mean']))
print('Relative variance of Y (IS):', planar.sci(RelVar_ismc))
print('Relative variance of Z (IS):', planar.sci(RelVar_ismc/result_ismc['niter']))
print('Process time (sec):', round(result_ismc['time'], 3))
print('Number of iterations:', result_ismc['niter'])
