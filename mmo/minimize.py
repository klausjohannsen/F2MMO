#!/usr/bin/env python

# std libs
import numpy as np
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy as copy
import matplotlib.pyplot as plt
from sklearn import linear_model
import mmo
import random

###############################################################################
# classes
###############################################################################
class Cloud:
    def __init__(self, ex, ey, x, distances):
        self.x = x
        self.ex = ex
        self.ey = ey
        self.distances = distances
        self.distance = np.median(distances)

class Clouds:
    def __init__(self, ex, ey, n, x):
        self.ex = ex
        self.ey = ey
        self.x = x
        self.nbrs = NearestNeighbors(n_neighbors = n, algorithm = 'ball_tree').fit(self.ex)
        self.distances, self.indices = self.nbrs.kneighbors(x)

    def __iter__(self):
        self.iter = 0
        return(self)

    def __next__(self):
        if self.iter == self.x.shape[0]:
            raise StopIteration
        c = Cloud(self.ex[self.indices[self.iter]], self.ey[self.indices[self.iter]], self.x[self.iter], self.distances[self.iter])
        self.iter += 1
        return(c)

class Cpcma:
    def __init__(self, f = None, domain = None, verbose = 0, budget = np.inf, max_iter = 10**20):
        assert(f is not None)
        assert(domain is not None)
        self.f = f
        self.domain = domain
        self.dim = len(self.domain[0])
        self.budget = budget
        self.max_iter = max_iter
        self.verbose_1 = verbose >= 1
        self.verbose_2 = verbose >= 2
        self.verbose_3 = verbose >= 3
        self.n_fct_calls = 0
        self.n_local_solves = 0
        self.solutions = np.zeros((0, self.dim))
        self.ll = np.array(self.domain[0], dtype = float)
        self.ur = np.array(self.domain[1], dtype = float)
        self.ex = np.zeros((0, self.dim))
        self.ey = np.zeros(0)
        self.l0 = np.min(self.ur - self.ll)

    def score(self, x, n = 10):

        m1 = np.zeros(x.shape[0])
        m2 = np.zeros(x.shape[0])
        for k, c in enumerate(Clouds(self.ex, self.ey, n, x)):
            m1[k] = c.distance
            clf = linear_model.LinearRegression()
            clf.fit(c.ex, c.ey)
            y = clf.predict(c.ex)
            m2[k] = la.norm(c.ey - y) / la.norm(c.ey)

        return(m1 * m2)

    def fct(self, x):
        self.n_fct_calls += 1
        return(self.f(x))

    def __iter__(self):
        self.iter = 0
        self.sigma = la.norm(self.ur - self.ll) * 0.01
        self.x0 = 0.5 * (self.ll + self.ur)
        return(self)

    def __str__(self):
        s = ''
        if self.verbose_1:
            s += '## MultiModalMinimizer\n'
            s += f'll: {self.ll}\n'
            s += f'ur: {self.ur}\n'
            s += f'iteration: {self.iter - 1}\n'
            s += f'n_local_solves: {self.n_local_solves}\n'
            s += f'n_fct_calls: {self.n_fct_calls}\n'
            s += f'n_solutions: {self.solutions.shape[0]}\n'
        return(s)

    def __next__(self):
        # search 
        cma = mmo.Cma(f = self.fct, x0 = self.x0, sigma = self.sigma)
        self.n_local_solves += 1
        self.solutions = np.vstack((self.solutions, cma.x))
        self.ex = np.vstack((self.ex, cma.ex))
        self.ey = np.hstack((self.ey, cma.ey))
        x = self.ll.reshape(1, self.dim) + (self.ur - self.ll).reshape(1, self.dim) * np.random.rand(1000, self.dim)
        z = self.score(x)
        self.x0 = x[np.argmax(z)]

        # stop
        if self.n_fct_calls >= self.budget or self.iter >= self.max_iter:
            raise StopIteration

        # admin
        self.iter += 1
        return(copy(self))

    def plot(self, x = None):
        if self.dim != 2:
            return
        plt.xlim(self.ll[0], self.ur[0])
        plt.ylim(self.ll[1], self.ur[1])
        if self.ex.shape[0] > 0:
            plt.scatter(self.ex[:, 0], self.ex[:, 1], c = 'black', s = 3)
        if x is not None:
            plt.scatter(x[:,0], x[:,1], c = 'orange', s = 50)
        if self.solutions.shape[0] > 0:
            plt.scatter(self.solutions[:, 0], self.solutions[:, 1], c = 'green', s = 10)
        plt.show()

    def plot_with_score(self, x = None, n = 100):
        if self.dim != 2:
            return
        xx, yy = np.meshgrid(np.linspace(self.ll[0], self.ur[0], n), np.linspace(self.ll[1], self.ur[1], n))
        xv = xx.reshape(-1,1)
        yv = yy.reshape(-1,1)
        p = np.hstack((xv,yv))
        z = self.score(p)
        zz = z.reshape(n ,n)
        plt.figure(figsize=(9, 9))
        plt.pcolor(xx, yy, zz)
        #if self.ex.shape[0] > 0:
        #    plt.scatter(self.ex[:, 0], self.ex[:, 1], c = 'white', s = 3)
        if x is not None:
            plt.scatter(x[:,0], x[:,1], c = 'orange', s = 50)
        if self.solutions.shape[0] > 0:
            plt.scatter(self.solutions[:, 0], self.solutions[:, 1], c = 'green', s = 10)
        plt.scatter(self.x0[0], self.x0[1], c = 'red', s = 50)
        plt.xlim(self.ll[0], self.ur[0])
        plt.ylim(self.ll[1], self.ur[1])
        plt.show()


###############################################################################
# classes
###############################################################################
class Bscma:
    def __init__(self, f = None, domain = None, verbose = 0, budget = np.inf, max_iter = 10**20):
        assert(f is not None)
        assert(domain is not None)
        self.f = f
        self.domain = domain
        self.dim = len(self.domain[0])
        self.budget = budget
        self.max_iter = max_iter
        self.verbose_1 = verbose >= 1
        self.verbose_2 = verbose >= 2
        self.verbose_3 = verbose >= 3
        self.n_fct_calls = 0
        self.n_local_solves = 0
        self.solutions = np.zeros((0, self.dim))
        self.ll = np.array(self.domain[0], dtype = float)
        self.ur = np.array(self.domain[1], dtype = float)
        self.ex = np.zeros((0, self.dim))
        self.ey = np.zeros(0)
        self.l0 = np.min(self.ur - self.ll)

    def fct(self, x):
        self.n_fct_calls += 1
        return(self.f(x))

    def __iter__(self):
        self.iter = 0
        self.sigma = la.norm(self.ur - self.ll) * 0.1
        self.x0 = 0.5 * (self.ll + self.ur)
        return(self)

    def __str__(self):
        s = ''
        if self.verbose_1:
            s += '## Bisection MultiModalMinimizer\n'
            s += f'll: {self.ll}\n'
            s += f'ur: {self.ur}\n'
            s += f'iteration: {self.iter - 1}\n'
            s += f'n_local_solves: {self.n_local_solves}\n'
            s += f'n_fct_calls: {self.n_fct_calls}\n'
            s += f'n_solutions: {self.solutions.shape[0]}\n'
        return(s)

    def __next__(self):
        # search 
        cma = mmo.Cma(f = self.fct, x0 = self.x0, sigma = self.sigma)
        self.n_local_solves += 1
        self.solutions = np.vstack((self.solutions, cma.x))
        self.ex = np.vstack((self.ex, cma.ex))
        self.ey = np.hstack((self.ey, cma.ey))

        # score obtained by bisection
        self.bt = mmo.BinaryTree(domain = self.domain, xy = [self.ex, self.ey])
        #self.bt.plot(p = [(solutions, 'orange', 50), (m.solutions, 'green', 20)])
        self.x0, self.sigma = self.bt.seed()

        # stop
        if self.n_fct_calls >= self.budget or self.iter >= self.max_iter:
            raise StopIteration

        # admin
        self.iter += 1
        return(copy(self))









