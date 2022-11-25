#!/usr/bin/env python

# libraries
import numpy as np
import numpy.linalg as la
import mmo 
from cec2019comp100digit import cec2019comp100digit
import sys

####################################################################################################
# config
####################################################################################################
FCT = int(sys.argv[1])
BUDGET = 120000000

####################################################################################################
# solutions, objective function and domain
####################################################################################################
DIM = 10
LEN = 100
if FCT == 1: 
    DIM = 9
    LEN = 8192
if FCT == 2: 
    DIM = 16
    LEN = 16384
if FCT == 3: 
    DIM = 18
    LEN = 4

bench = cec2019comp100digit
bench.init(FCT, DIM) 
domain = [ [-LEN]*DIM, [LEN]*DIM ] 

####################################################################################################
# run
####################################################################################################
def digits(mmm):
    y = np.zeros(mmm.solutions.shape[0])
    for k in range(mmm.solutions.shape[0]):
        y[k] = bench.eval(mmm.solutions[k])
    y_min = np.min(y)
    print(f'  min = {y_min:.10e}')
    d = y_min - 1
    digits = 0
    for k in range(10):
        if d < 10 ** (-k):
            digits = k + 1
        else:
            break
    print(f'  digits = {digits}')
    return(digits)

####################################################################################################
# run
####################################################################################################
minimizer = mmo.Bscma(f = bench.eval, domain = domain, verbose = 1, budget = BUDGET)
for k, m in enumerate(minimizer):
    print("############################")
    print(m.bt)
    print(m)
    m.bt.plot(p = [(m.ex, 'white', 1), (m.solutions, 'green', 20), (m.x0.reshape(1, -1), 'red', 50)])

    if digits(m) >= 10:
        exit()

m.bt.plot(p = [(m.ex, 'white', 1), (m.solutions, 'green', 20), (m.x0.reshape(1, -1), 'red', 50)])

digits(m)







