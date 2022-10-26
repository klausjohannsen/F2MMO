#!/usr/bin/env python

# libraries
import numpy as np
import numpy.linalg as la
import mmo 
from modules_cec13.cec2013 import how_many_goptima, CEC2013
#from hyperqueue import Job, LocalCluster
from time import time
import sys

####################################################################################################
# config
####################################################################################################
PROBLEM = int(sys.argv[1])

####################################################################################################
# objective function and domain
####################################################################################################
cec = CEC2013(PROBLEM)
DIM = cec.get_dimension()
N_SOL = cec.get_no_goptima()
BUDGET = cec.get_maxfes()
OPTIMUM = cec.get_fitness_goptima()

# optima
if PROBLEM > 10:
    solutions = np.loadtxt('modules_cec13/data/optima.dat')
    solutions = solutions[:N_SOL, :DIM]
else:
    solutions = np.loadtxt(f'modules_cec13/data/optima_problem_{PROBLEM}')
    if len(solutions.shape) == 1:
        solutions = solutions.reshape(1, -1)
    solutions = solutions[:, :DIM]

# function and domain
LL = []
UR = []
for d in range(DIM):
    LL += [cec.get_lbound(d)]
    UR += [cec.get_ubound(d)]
f = lambda x: -cec.evaluate(x)
domain = [LL, UR]

print("## configuration")
print(f'PROBLEM = {PROBLEM}')
print(f'DIM = {DIM}')
print(f'N_SOL = {N_SOL}')
print(f'BUDGET = {BUDGET}')
print(f'OPTIMUM = {OPTIMUM}')
print()

####################################################################################################
# fct
####################################################################################################
def n_solutions(x):
    CEC_f = CEC2013(PROBLEM)
    n_optima = CEC_f.get_no_goptima()
    count = np.zeros((5))
    count[0], seeds = how_many_goptima(x, CEC_f, 1e-1)
    count[1], seeds = how_many_goptima(x, CEC_f, 1e-2)
    count[2], seeds = how_many_goptima(x, CEC_f, 1e-3)
    count[3], seeds = how_many_goptima(x, CEC_f, 1e-4)
    count[4], seeds = how_many_goptima(x, CEC_f, 1e-5)
    n = np.mean(count)
    return(n)

####################################################################################################
# run
####################################################################################################
minimizer = mmo.Bscma(f = f, domain = domain, verbose = 1, budget = BUDGET)
for k, m in enumerate(minimizer):
    print("############################")
    print(m.bt)
    print(m)
    n = n_solutions(m.solutions)
    print(f'n_solutions: {n}')
    if n == N_SOL:
        break
    if k % 10 == 0:
        m.bt.plot(p = [(m.ex, 'white', 1), (solutions, 'orange', 50), (m.solutions, 'green', 20), (m.x0.reshape(1, -1), 'red', 50)])

m.bt.plot(p = [(m.ex, 'white', 1), (solutions, 'orange', 50), (m.solutions, 'green', 20), (m.x0.reshape(1, -1), 'red', 50)])

####################################################################################################
# peak rate
####################################################################################################
CEC_f = CEC2013(PROBLEM)
n_optima = CEC_f.get_no_goptima()
count = np.zeros((5))
count[0], seeds = how_many_goptima(m.solutions, CEC_f, 1e-1)
count[1], seeds = how_many_goptima(m.solutions, CEC_f, 1e-2)
count[2], seeds = how_many_goptima(m.solutions, CEC_f, 1e-3)
count[3], seeds = how_many_goptima(m.solutions, CEC_f, 1e-4)
count[4], seeds = how_many_goptima(m.solutions, CEC_f, 1e-5)
peake_rate = np.mean(count) / n_optima

print("#####################")
print("## N solutions")
print("#####################")
print(PROBLEM, peake_rate)
print()







