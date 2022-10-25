import numpy as np
import numpy.linalg as la
from sklearn import linear_model
from copy import deepcopy as copy
from matplotlib import pyplot as plt, patches
import matplotlib.colors as colors
import matplotlib.cm as cmx

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
    def __init__(self, ll = None, ur = None, xy = None):
        self.ll = ll
        self.ur = ur
        self.dim = self.ll.shape[0]
        self.xy = xy
        self.rle = linear_eval(self.xy)
        self.logvolume = np.sum(np.log(self.ur - self.ll))
        self.n1 = None
        self.n2 = None
        if self.dim == 2:
            self.score = np.exp(self.logvolume + np.log(self.rle))

    def __str__(self):
        s = '# node\n'
        s += f'll = {self.ll}\n'
        s += f'ur = {self.ur}\n'
        s += f'data = {self.xy[0].shape}\n'
        s += f'rle = {self.rle}\n'
        s += f'leaf = {self.n1 == None}'
        return(s)

    def split_(self):
        assert(self.n1 == None)
        assert(self.n2 == None)
        x = self.xy[0]
        y = self.xy[1]
        axis = np.argmax(self.ur - self.ll)
        x_median = np.median(x[:, axis])
        ll_1 = copy(self.ll)
        ur_1 = copy(self.ur)
        ll_2 = copy(self.ll)
        ur_2 = copy(self.ur)
        ur_1[axis] = x_median
        ll_2[axis] = x_median
        isin_1 = np.zeros(x.shape[0], dtype = bool)
        isin_2 = np.zeros(x.shape[0], dtype = bool)
        for k in range(x.shape[0]):
            if np.all(x[k] >= ll_1) and np.all(x[k] < ur_1):
                isin_1[k] = True
            if np.all(x[k] >= ll_2) and np.all(x[k] < ur_2):
                isin_2[k] = True
        if np.sum(isin_1) < 5 * self.dim:
            return(False)
        if np.sum(isin_2) < 5 * self.dim:
            return(False)
        self.n1 = Node(ll = ll_1, ur = ur_1, xy = [x[isin_1], y[isin_1]])
        self.n2 = Node(ll = ll_2, ur = ur_2, xy = [x[isin_2], y[isin_2]])
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
    def __init__(self, domain = None, xy = None):
        self.domain = domain
        self.xy = xy
        self.ll = np.array(self.domain[0], dtype = float)
        self.ur = np.array(self.domain[1], dtype = float)
        self.dim = self.ll.shape[0]
        self.root = Node(ll = self.ll, ur = self.ur, xy = self.xy)
        self.root.split()

    def leaf_nodes(self):
        return(self.root.leaf_nodes())

    def seed(self):
        n_best = None
        best_score = -np.inf
        for n in self.root.leaf_nodes():
            if n.score > best_score:
                n_best = n
                best_score = n.score
        sigma = 0.1 * np.min(n_best.ur - n_best.ll)
        x0 = 0.5 * (n_best.ll + n_best.ur)
        return(x0, sigma)

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
        plt.colorbar(scalarMap, ax = ax)
        plt.show()












