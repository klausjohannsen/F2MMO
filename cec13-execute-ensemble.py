# libs
import numpy as np
from modules import hq_run_n

# fct
def geo_mean(x):
    r = np.exp(np.mean(np.log(x)))
    return(r)

# run
#for PROBLEM in range(1, 21):
for PROBLEM in [6, 8, 9]:
    r = hq_run_n(cmd = f'python3 cec13-execute-p1-20.py {PROBLEM}', n = 50, pp = 'last_2_float')
    print(f'n runs = {len(r)}')
    r = np.array(r)
    print(f'PROBLEM: {np.mean(r[:, 0])}')
    print(f'PEAK RATE: {np.mean(r[:, 1])}')
    print()

    with open("results/results-cec13.txt", "a") as f:
        f.write(f'PROBLEM: {np.mean(r[:, 0])}, PEAK RATE: {np.mean(r[:, 1])}\n')
    





