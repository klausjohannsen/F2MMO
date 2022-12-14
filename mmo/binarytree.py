import numpy as np
import numpy.linalg as la
from sklearn import linear_model
from copy import deepcopy as copy
from matplotlib import pyplot as plt, patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.cluster.vq import kmeans, vq

# fcts
def linear_eval(xy):
    x = xy[0]
    y = xy[1]
    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    yy = clf.predict(x)
    #r = la.norm(y - yy) / la.norm(y)
    r = np.max(np.abs(y - yy)) / np.max(np.abs(y))
    return(r)

# classes
class Node:
    def __init__(self, ll = None, ur = None, xy = None, score_limit = 0):
        self.ll = ll
        self.ur = ur
        self.dim = self.ll.shape[0]
        self.xy = xy
        self.rle = linear_eval(self.xy)
        self.volume = np.prod(self.ur - self.ll)
        self.n1 = None
        self.n2 = None
        self.score = self.volume * self.rle
        if score_limit >= 0:
            self.score_limit = score_limit
        else:
            self.score_limit = - score_limit * self.score

    def __str__(self):
        s = '# node\n'
        s += f'll = {self.ll}\n'
        s += f'ur = {self.ur}\n'
        s += f'data = {self.xy[0].shape}\n'
        s += f'rle = {self.rle}\n'
        s += f'leaf = {self.n1 == None}'
        return(s)

    def split_parameters(self):
        x = self.xy[0]
        axis = np.argmax(self.ur - self.ll)
        x_median = np.median(x[:, axis])
        eta = (x_median - self.ll[axis]) / (self.ur[axis] - self.ll[axis])
        return(axis, x_median)

    def split_(self):
        if self.score < self.score_limit:
            return(False)
        assert(self.n1 == None)
        assert(self.n2 == None)
        x = self.xy[0]
        y = self.xy[1]
        axis, x_div = self.split_parameters()
        ll_1 = copy(self.ll)
        ur_1 = copy(self.ur)
        ll_2 = copy(self.ll)
        ur_2 = copy(self.ur)
        ur_1[axis] = x_div
        ll_2[axis] = x_div
        isin_1 = np.all(np.logical_and(x >= ll_1.reshape(1, -1), x < ur_1.reshape(1, -1)), axis = 1)
        isin_2 = np.all(np.logical_and(x >= ll_2.reshape(1, -1), x < ur_2.reshape(1, -1)), axis = 1)
        if np.sum(isin_1) < 5 * self.dim:
            return(False)
        if np.sum(isin_2) < 5 * self.dim:
            return(False)
        self.n1 = Node(ll = ll_1, ur = ur_1, xy = [x[isin_1], y[isin_1]], score_limit = self.score_limit)
        self.n2 = Node(ll = ll_2, ur = ur_2, xy = [x[isin_2], y[isin_2]], score_limit = self.score_limit)
        return(True)

    def split(self):
        if self.split_() == True:
            self.n1.split()
            self.n2.split()

    def leaf_nodes(self):
        if self.n1 == None:
            return([self])
        else:
            lnl_1 = self.n1.leaf_nodes()
            lnl_2 = self.n2.leaf_nodes()
            return(lnl_1 + lnl_2)

class BinaryTree:
    def __init__(self, domain = None, xy = None, score_limit = -0.1):
        self.domain = domain
        self.xy = xy
        self.ll = np.array(self.domain[0], dtype = float)
        self.ur = np.array(self.domain[1], dtype = float)
        self.dim = self.ll.shape[0]
        self.score_limit = score_limit
        self.root = Node(ll = self.ll, ur = self.ur, xy = self.xy, score_limit = self.score_limit)
        self.root.split()

    def __str__(self):
        s = '# binary tree\n'
        s += f'll = {self.ll}\n'
        s += f'ur = {self.ur}\n'
        s += f'dim = {self.dim}\n'
        s += f'data = {self.xy[0].shape}\n'
        s += f'n_leaf = {len(self.leaf_nodes())}\n'
        s += f'score_limit = {self.score_limit}\n'
        return(s)

    def leaf_nodes(self):
        return(self.root.leaf_nodes())

    def seed(self, bounds = False):
        n_best = None
        best_score = -np.inf
        for n in self.root.leaf_nodes():
            if n.score > best_score:
                n_best = n
                best_score = n.score
        C = 0.333 * np.diag(n_best.ur - n_best.ll)
        x0 = 0.5 * (n_best.ll + n_best.ur)
        r_bounds = None
        if bounds:
            r_bounds = np.vstack((n_best.ll, n_best.ur)).T
        return(x0, C, r_bounds)

    def plot(self, p = None):
        if self.dim != 2:
            return
        ln = self.leaf_nodes()
        plt.rcParams["figure.figsize"] = [11.3, 9]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cm = plt.get_cmap('jet')
        vmin = np.inf
        vmax = -np.inf
        for n in ln:
            vmin = min(vmin, n.score)
            vmax = max(vmax, n.score)
        cNorm  = colors.Normalize(vmin = vmin, vmax = vmax)
        scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cm)
        for n in ln:
            colorVal = scalarMap.to_rgba(n.score)
            rectangle = patches.Rectangle(n.ll, n.ur[0] - n.ll[0], n.ur[1] - n.ll[1], edgecolor = 'black', facecolor = colorVal, linewidth = 1)
            ax.add_patch(rectangle)
        for pp in p:
            x = pp[0]
            c = pp[1]
            s = pp[2]
            plt.scatter(x[:,0], x[:,1], c = c, s = s)
        plt.xlim(self.ll[0], self.ur[0])
        plt.ylim(self.ll[1], self.ur[1])
        plt.colorbar(scalarMap, orientation="vertical")
        plt.show()












