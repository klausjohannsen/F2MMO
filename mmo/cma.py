import numpy as np
import numpy.linalg as la
from cmaes import CMA

# fcts
def moving_average__(a, n = 10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_average(a, n = 10):
    print(a.shape)
    x = []
    for k in range(a.shape[1]):
        x += [moving_average__(a[:, k], n = n)]
    x = np.vstack(x).T
    return(x)

# cma
class Cma:
    def __init__(self, f = None, x0 = None, sigma = None, max_gen = 10**20):
        optimizer = CMA(mean = x0, sigma = sigma)
        y_best = np.inf
        n_fct_eval = 0
        ex = []
        ey = []
        for gen in range(max_gen):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                y = f(x)
                n_fct_eval += 1
                solutions.append((x, y))
                if y < y_best:
                    x_best = x
                    y_best = y
            optimizer.tell(solutions)
            ex += [s[0] for s in solutions]
            ey += [s[1] for s in solutions]
            if optimizer.should_stop():
                break

        self.n_fct_eval = n_fct_eval
        self.x = x_best
        self.y = y_best
        self.n_gen = gen
        self.ex = np.vstack(ex)
        self.ey = np.hstack(ey)

    def __str__(self):
        s = '## cmaes\n'
        s += f'x = {self.x}\n'
        s += f'y = {self.y}\n'
        s += f'n_gen = {self.n_gen}\n'
        s += f'n_fct_eval = {self.n_fct_eval}'
        s += f'n_path = {self.path.shape[0]}'
        return(s)







