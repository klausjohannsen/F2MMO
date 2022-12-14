# libs
import numpy as np
import numpy.linalg as la
import mmo
from scipy.spatial import distance_matrix
from modules import t

####################################################################################################
# config
####################################################################################################
N_SOL = 4
DIM = 2 
BUDGET = 1000000

####################################################################################################
# solutions, objective function and domain
####################################################################################################
solutions = 0.1 + 0.8 * np.random.rand(N_SOL, DIM)

def solutions_found(x):
    dm = distance_matrix(x, solutions)
    dm = np.min(dm, axis = 0)
    n = np.sum(dm < 1e-8)
    return(n)

def f(x):
    r = la.norm(solutions - x.reshape(1, -1), axis = 1)
    return(np.min(r) ** 2)

domain = [[0]*DIM, [1]*DIM]

####################################################################################################
# run
####################################################################################################
minimizer = mmo.Bscma(f = f, domain = domain, verbose = 1)
for m in minimizer:
    print(m)
    print()
    print(m.bt)
    print()

    n = solutions_found(m.solutions)
    print(f'solutions found: {n}')

    m.bt.plot(p = [(m.ex, 'white', 1), (solutions, 'orange', 50), (m.solutions, 'green', 20), (m.x0.reshape(1, -1), 'red', 50)])

    print()
    if n == N_SOL:
        break
    
m.bt.plot(p = [(solutions, 'orange', 50), (m.solutions, 'green', 20), (m.x0.reshape(1, -1), 'red', 50)])


