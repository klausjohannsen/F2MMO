# libs
import numpy as np
import numpy.linalg as la
import mmo
from scipy.spatial import distance_matrix

####################################################################################################
# config
####################################################################################################
N_SOL = 50
DIM = 2 
BUDGET = 1000000

####################################################################################################
# solutions, objective function and domain
####################################################################################################
solutions = 0.1 + 0.1 * np.random.rand(N_SOL, DIM)

def solutions_found(x):
    dm = distance_matrix(x, solutions)
    dm = np.min(dm, axis = 0)
    n = np.sum(dm < 1e-8)
    return(n)

def f(x):
    r = la.norm(solutions - x.reshape(1, -1), axis = 1)
    return(np.min(r) ** 0.5)

domain = [[0]*DIM, [1]*DIM]

####################################################################################################
# run
####################################################################################################
minimizer = mmo.Cpcma(f = f, domain = domain, verbose = 1)
for m in minimizer:
    print(m)
    print()

    n = solutions_found(m.solutions)
    print(f'solutions found: {n}')
    if n % 500 == 0:
        m.plot_with_score(x = solutions)

    print()
    if n == N_SOL:
        break
    
m.plot_with_score(x = solutions)



