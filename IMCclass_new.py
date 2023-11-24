#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:11:53 2022
Updated on Sat Apr 08 08:17:51 2023

@author: ym
"""

import Library as lib
import math
import itertools
import time
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import erf
from collections import deque

"""
IMC (abstraction) class passes in 
state space X as boxes, e.g. X = [[lb1, up1], [lb2, up2], ...];
precision (completeness parameter) eps;
f in the dynamics X_{t+1} = f(X_t) + b(X_t)w_t;
b in the dynamics X_{t+1} = f(X_t) + b(X_t)w_t;

We use the notations in the paper
'Robustly Complete Finite-State Abstractions 
for Verification of Stochastic Systems', FORMATS 2022.
"""

class IMC():
    #The coefficient used to generate a pseudo-discretization of the state 
    #space during the generation of the image of inclusion functions, 
    #such that the distance between the image of the inclusion functions 
    #and the actual image of the nonlinear vector field can be bounded by 
    #the prescibed precision times the kappa_coefficient.
    kappa_coefficient = 0.50    
    
    #Set a ratio that is within (0, 1) for the prescibed precision
    eps_margin = 0.999
    
    #The sup Lipschitz constant of test functions of ws norm.
    ball_coefficient = 1 
    
    #Specified ws distance ratio w.r.t. the grid size.
    #The multiplication of ws_dist_ratio and the grid size returns 
    #the distance w.r.t. the reference measure.
    ws_dist_ratio = 1.3  #0.25
    
    #error tolerance
    err = 0.00000001
    
    
    def __init__(self, state, precision, f, b, L_f, L_b, \
                 cor=None, fp=None, bp=None, use_fn_f=None, use_fn_b=None):
        self.dim = len(state)
        self.X = state
        self.diam = max(self.X[i][1]-self.X[i][0] for i in range(0, self.dim))
        self.vol = math.prod(self.X[i][1]-self.X[i][0] \
                             for i in range(0, self.dim))
        self.precision = precision
        
        #This is used for a direct calculation of eta
        self.eps = self.precision * (self.eps_margin - self.kappa_coefficient)
        
        #If f is constant, then there record it as a list. 
        self.f = f
        self.b = b if self.dim >= 2 else [[b]]
        self.L_f = L_f
        self.L_b = L_b if self.dim >= 2 else [[L_b]]
        self.L_b_max = np.array([L_b]).max()
        self.cor = cor
        self.fp = fp
        self.bp = bp  
        self.use_fn_f = use_fn_f
        self.use_fn_b = use_fn_b
        
        #Determine the 'inflation' precision for the inclusion functions
        self.kappa = self.kappa_coefficient * self.precision
        
        self.eta, self.N = self.__get_eta_N()
        self.N_matrix = math.prod(self.N) + 1
        self.w, self.N_incl = self.__get_w_N_grid()
        
        #__getQ() returns an iterable 
        #__getQ() works faster than using np.mgrid + np.dstack for high dim
        #Q iterates the 'floor' points of ''state'' grids
        self.Q = self.getQ
              
        #get the size beta of grids for mean and covariance
        self.beta = self.ws_dist_ratio * self.eta if self.L_b == 0 \
            else self.ws_dist_ratio * self.eta / math.sqrt(2)
        
    #Evaluate a 1-dim Gaussian distribution from -infty to x 
    #using the antiderivatie
    @staticmethod
    def Phi(x, mean, sig):
        if sig != 0:
            return 0.5*(math.erf((x - mean) / (sig * math.sqrt(2))))+0.5
            #return 0.5*(erf((x - mean) / (sig * math.sqrt(2))))+0.5
        else:
            return int(x > mean)
    
    @classmethod
    def chkappa_coefficient(cls, new_coefficient):
        cls.kappa_coefficient = new_coefficient    
    
    @classmethod
    def cheps_margin(cls, new_coefficient):
        cls.eps_margin = new_coefficient  
        
    @classmethod
    def chball_coefficient(cls, new_coefficient):
        cls.ball_coefficient = new_coefficient 
        
    @classmethod
    def chws_dist(cls, new_coefficient):
        cls.ws_dist_ratio = new_coefficient 
        
    @classmethod
    def cherr(cls, new_coefficient):
        cls.err = new_coefficient  
        
    @property 
    def dictionary(self):
        return {'dimension_of_workplace': [self.dim],\
               'diam_of_workplace': [self.diam], \
               'vol_of_workplace': [self.vol], \
               'number_of_grid_workplace': [self.N], \
               'dimension_of_IMC': [self.N_matrix], \
                'gridsize_of_workplace': [self.eta], \
                'gridsize_of_measure': [self.beta]}
    
    #round up the number of grids of the state space
    def __adjust_N(self, size):
        for i in range(0, self.dim):
            yield round(math.floor((self.X[i][1] - self.X[i][0]) / size) + 1)
            
    #round up the number of grids for discretizing 
    #the domain of inclusion functions
    def __adjust_N_grid(self, size):
        for i in range(0, self.dim):
            yield round(math.floor(self.eta / size) + 1)
    
    
    #use the complete analysis to obtain the grid size eta (in Eq.(19))
    def __get_eta_N(self): 
        eta = self.eps / (self.ws_dist_ratio + 2 * self.ball_coefficient)
        N = self.__adjust_N(eta)
        eta_refine = (self.vol / math.prod(self.__adjust_N(eta))) \
            ** (1 / self.dim)
        return eta_refine, list(N)
    
    #Generate the discretized grids.
    @property
    def getQ(self):
        def Qgenerator(self):
            for i in range(0, self.dim):
                yield np.linspace(self.X[i][0], self.X[i][1], \
                                  self.N[i]).tolist()
        return itertools.product(*Qgenerator(self))   
    
    
    #Get the size of [y], i.e. w([y]),
    #for inclusion functions on each grid.
    def __get_w_N_grid(self):
        vol = math.prod(self.eta for i in range(0, self.dim))
        
        #Decide if the size of a state grid, eta,
        #is already small enough to make 
        #the set of Gauss measures within the required Wasserstein distance, 
        #which in this case is self.kappa.
        #Usually, when L_f**2 + L_b_max**2 is large, one needs to further split
        #the grid into [y], and patch up [mean] and [var]; 
        #otherwise we just use the state grid to generate [mean] and [var].
        
        #Note that when L_f**2 + L_b_max**2 = 0, 
        #[mean] and [var] reduce to singletons. 
        
        #This function returns conservative w and a discretization for the 
        #generation of inclusion of the 'reachable set of Gauss measures'. 
        
        if self.L_f**2 + self.L_b_max**2 != 0:
            w = math.sqrt((self.ball_coefficient * self.kappa)**2 \
                                / (self.L_f**2 + self.L_b_max**2))
        else:
            w = self.eta
        w_input = self.eta + 0.000001 if w >= self.eta else w
        N = self.__adjust_N_grid(w_input)
        w_refine = (vol / math.prod(self.__adjust_N_grid(w))) ** (1 / self.dim)
        print('w_refine=', w_refine)
        print('N=', list(self.__adjust_N_grid(w_input)))
        return w_refine, list(N)
    
    #Generate [y], which is a pseudo-grid to generate the inclusion of the
    #'reachable set of Gauss measures'. 
    #As a continuation of __get_w_N_grid( ), 
    #this function records the 'bottom-left' point of the required-size grids.
    def __get_incl_meas_grid(self, grid):
        def Qgenerator(self):
            for i in range(0, self.dim):
                yield np.linspace(grid[i][0], grid[i][1], 
                                  self.N_incl[i]).tolist()
        return itertools.product(*Qgenerator(self))  
    
    #Generate [mean]. 
    #This is for storage purpose, hence 'yield' is preferred over 'return'.
    def __overapprx_mean_row(self, grid_point):
        grid = [[grid_point[i], grid_point[i] + self.eta] \
                for i in range(self.dim)]
        Q_meas = self.__get_incl_meas_grid(grid)
        for q in Q_meas:
            g_box = [[q[i], q[i] + self.w] for i in range(self.dim)]
            result = lib.fn(g_box, self.f) if self.use_fn_f is not None \
                else (lib.fc(g_box, self.f, self.L_f, self.fp) if self.L_f != 0 \
                      else self.f)
            yield result
            
    #Generate [std] (or [Cholesky]), which is supposed to be a dim * dim matrix.       
    #In this function, 
    #the dim * dim matrix is reshaped as a dim^2 * 1 list of Boxes.
    def __overapprx_std_row(self, grid_point):
        grid = [[grid_point[i], grid_point[i] + self.eta] \
                for i in range(self.dim)]
        Q_meas = self.__get_incl_meas_grid(grid)
        for q in Q_meas:
            g_box = [[q[i], q[i] + self.w] for i in range(self.dim)]
            result = [lib.fn([g_box[i]], self.b[i][j]) \
                      if self.use_fn_b is not None \
                      else(lib.fc([g_box[i]], self.b[i][j], self.L_b[i][j]) \
                           if self.L_b[i][j] != 0 else self.b[i][j]) \
                          for i in range(self.dim) for j in range(self.dim)]
            yield result
    
    #discretize [mean] by beta and find ref points of mean
    def __mean_ref(self, grid_point):
        #Generate ref point of mean.
        def mean_ref_generator(self):
            for mean_box in self.__overapprx_mean_row(grid_point):
                for i in range(self.dim):
                    if isinstance(mean_box, lib.Box):
                        a = math.floor(mean_box.X[i][1] / self.beta) * self.beta
                        yield np.mgrid[mean_box.X[i][0]:a:self.beta] \
                            if a >= mean_box.X[i][0] else [mean_box.X[i][0]]
                    else:
                        yield [mean_box[i]]
        return itertools.product(*mean_ref_generator(self)) 
    
    #discretize [std] (or [Cholesky]) by beta and find ref points of mean
    def __std_ref(self, grid_point):
        #Generate ref point of mean.
        def std_ref_generator(self):
            for std_box in self.__overapprx_std_row(grid_point):
                for std in std_box:
                    if isinstance(std, lib.Box):
                        a = math.floor(std.X[0][1] / self.beta) * self.beta
                        yield np.mgrid[std.X[0][0]:a:self.beta]  \
                            if a >= std.X[0][0] else [std.X[0][0]]
                    else:
                        #In this case, the current entry of b is a constant.
                        #The generator should yield an iterable 
                        #rather than a number for the downstream calculation.
                        yield [std]
                        #yield [math.floor(std / self.beta) * self.beta]
        return itertools.product(*std_ref_generator(self))  
    
        
    def __evaluate_discrete_probability(self, mean, std, q):
        #Evaluate the transition probability T(grid_point, q).
        
        #For each q, 
        #we generate the refs of transition measure T(grid_point) (Gaussians).
        
        #The refs should be duplicated for each q.
        
        #For dim == 2, 
        #direct multiplication of antiderivative is faster than math.prod.
        
        #For L_b != 0, the evaluation depends on the correlation:
        #if each dimension is independent, 
        #multiplication of antiderivative is 100 times 
        #faster than using scipy cdf. 
        
        if self.dim == 2 and not self.cor:
            return (self.Phi(q[0] + self.eta, mean[0], std[0][0])             \
                            - self.Phi(q[0], mean[0], std[0][0]))             \
                     * (self.Phi(q[1] + self.eta, mean[1], std[1][1])         \
                            - self.Phi(q[1], mean[1], std[1][1]))
        elif self.dim != 2 and not self.cor:       
            return math.prod((self.Phi(q[i] + self.eta, mean[i], std[i][i])   \
                                - self.Phi(q[i], mean[i], std[i][i]))         \
                                           for i in range(self.dim))
        elif self.dim == 2 and self.cor:
            var = np.dot(std, std.T)
            rv = multivariate_normal(mean, var)
            return rv.cdf((q[0] + self.eta, q[1] + self.eta))                 \
                                     + rv.cdf((q[0], q[1]))                   \
                                     - rv.cdf((q[0], q[1] + self.eta))        \
                                     - rv.cdf((q[0] + self.eta, q[1]))
        else:
            var = np.dot(std, std.T)
            rv = multivariate_normal(mean, var)
            I = itertools.product(*((-1, 1) for i in range(self.dim)))
            J = itertools.product(*((q[i], q[i] + self.eta) \
                                        for i in range(self.dim)))
            arr = np.array([rv.cdf(j) * math.prod(i) for i, j in zip(I, J)])    
            return np.sum(arr)
            
    def __evaluate_cemetary_probability(self, mean, std):
        if self.dim == 2 and not self.cor:
            return 1 - (self.Phi(self.X[0][1], mean[0], std[0][0])            \
                         - self.Phi(self.X[0][0], mean[0], std[0][0]))        \
                         * (self.Phi(self.X[1][1], mean[1], std[1][1])        \
                         - self.Phi(self.X[1][0], mean[1], std[1][1]))
        elif self.dim != 2 and not self.cor:       
            return 1 - math.prod((self.Phi(self.X[i][1], mean[i], std[i][i])  \
                         - self.Phi(self.X[i][0], mean[i], std[i][i]))        \
                                            for i in range(self.dim))
                    
        elif self.dim == 2 and self.cor:
            var = np.dot(std, std.T)
            rv = multivariate_normal(mean, var)
            rv_cdf = rv.cdf((self.X[0][1], self.X[1][1]))                     \
                         + rv.cdf((self.X[0][0], self.X[1][0]))               \
                         - rv.cdf((self.X[0][0], self.X[1][1]))               \
                         - rv.cdf((self.X[0][1], self.X[1][0]))
            return 1 - rv_cdf
                    
        else:
            var = np.dot(std, std.T)
            rv = multivariate_normal(mean, var)
            I = itertools.product(*((-1, 1) for i in range(self.dim)))
            J = itertools.product(*((self.X[i][0], self.X[i][1]) \
                                        for i in range(self.dim)))
            arr = np.array([rv.cdf(j) * math.prod(i) for i, j in zip(I, J)])             
            return 1 - np.sum(arr)     
        
    #For each ref mean and ref var, generate a ref row measure.
    def __get_row_measure(self, grid_point):
        for mean in self.__mean_ref(grid_point):  
            for std_ref in self.__std_ref(grid_point):
                std = np.array(std_ref).reshape(np.array(self.b).shape)
                std = np.abs(std)
                row_measure_grid = \
                    np.array([self.__evaluate_discrete_probability(mean, std, \
                                                    q) for q in self.getQ])
                row_measure_cemetary = \
                    np.array([self.__evaluate_cemetary_probability(mean, std)])
                
                row_measure = \
                    np.hstack((row_measure_grid, row_measure_cemetary))
                row_measure = np.where(row_measure < self.err, 0, row_measure)
                yield row_measure / np.sum(row_measure) #normalize
                
    def __get_row_ref_measure(self, grid_point):
        mean = next(self.__mean_ref(grid_point))
        std_ref = next(self.__std_ref(grid_point))
        std_reshape = np.array(std_ref).reshape(np.array(self.b).shape)
        std = self.b if self.L_b == 0 else std_reshape
        row_ref_measure_grid = \
            np.array([self.__evaluate_discrete_probability(mean, std, \
                                            q) for q in self.getQ])
        row_ref_measure_cemetary = \
            np.array([self.__evaluate_cemetary_probability(mean, std)])
        row_measure = \
            np.hstack((row_ref_measure_grid, row_ref_measure_cemetary))
        return row_measure / np.sum(row_measure)
    
    #Generate the entries of IMC
    def __bounds_row_measure(self, grid_point):
        upper_bounds = self.beta + \
            np.maximum.reduce([arr for arr in self.__get_row_measure(grid_point)])
        lower_bounds = -self.beta + \
            np.minimum.reduce([arr for arr in self.__get_row_measure(grid_point)])
        
        #add filters
        lower_bounds = np.where(lower_bounds < self.err, 0, lower_bounds)
        lower_bounds = \
            np.where(lower_bounds > 1 - self.err -self.beta, 1, lower_bounds)
        upper_bounds = np.where(upper_bounds > 1, 1, upper_bounds)
        upper_bounds = \
            np.where(upper_bounds < self.err + self.beta, 0, upper_bounds)
        bounds = list(zip(lower_bounds, upper_bounds))
        result = np.stack(bounds, axis=0)
        return result


    def __output(original_function):
        def wrapper_function(*args):
            count = 0
            for i, j in enumerate(original_function(*args)):
                if j[0] == 0 and j[1] == 0 :
                    continue
                else:
                    count += 1
                    yield i, j[0], j[1]
            yield count
        return wrapper_function
    
    #Get a row of IMC abstraction.
    #The entries whose lower bound and upper bound are both 0 are omitted.  
    @__output
    def getrow(self, grid_point):
        IMC_box = self.__bounds_row_measure(grid_point)
        assert IMC_box.shape[0] == self.N_matrix
        for i in range(self.N_matrix):
            yield IMC_box[i]
            
    def getrow_ref(self, grid_point):
        ref_measure = self.__get_row_ref_measure(grid_point)
        assert ref_measure.shape[0] == self.N_matrix
        nonzero_index = np.nonzero(ref_measure)
        for i in nonzero_index[0]:
            yield i, ref_measure[i]

        
    def getQ_slice(self, i, portion):
        slice_length = int(self.N_matrix / portion) + 1
        j = i * slice_length
        Q_iter = itertools.islice(self.getQ, j, j + slice_length)
        return j, Q_iter
        
    
    def get_transition_grid(self, q):
        return list(self.getrow(q))

    def iter_transition(self):
        for q in self.getQ:
            yield sum(1 for _ in self.getrow(q))
            
    def get_transition(self):
        count = np.fromiter(self.iter_transition(), dtype=int)
        return np.sum(count)





if __name__ == '__main__':
    
    """
    
    X = [[-1, 1]]
    precision = 0.1 #0.2
    
    L_f = 1
    L_b = 0
    cor = False

    f = lambda x: x
    b = 0

    imc = IMC(X, precision, f, b, L_f, L_b, cor)
    #imc.chws_dist(0.1)
    print(imc.ws_dist_ratio)
    print('diam, vol, eta, beta, N= ', 
          imc.diam, imc.vol, imc.eta, imc.beta, imc.N)
    
    slice_length = int(imc.N_matrix/32)
    Q = itertools.islice(imc.getQ, 32, 32+slice_length)
    tic = time.time()
    #c = imc.get_transition()
    #print('count=', c)
    print('calculating...')
    try:
        q = next(imc.getQ)
        q = next(imc.getQ)
        #q = next(Q)
        #q = next(Q)
        print(q)
        row = imc.getrow_ref(q)
        row2 = imc.getrow(q)
        
        for j in range(3000):
            #print(*next(row2))
            print(*next(row2))
            #print('intvl=', next(row))
        
    except StopIteration:
            print('Stopped at=', j)
    finally:
        print('Computation time of a single row is {} sec'.format(time.time()-tic))
        
        
    """
    
    X = [[-1, 1], [-1, 1]]
    precision = 0.08#0.08 #0.2
    
    L_f = 1
    L_b = [[0, 0, 0], [0, 0, 0]]
    cor = False

    f = lambda x: [x[0], x[1]]
    fp1 = lambda x: [1, 0]
    fp2 = lambda x: [0, 1]
    fp = [fp1, fp2]
    b = [[0.001, 0], [0, 0.001]]

    imc = IMC(X, precision, f, b, L_f, L_b, cor, fp)
    #imc.chws_dist(0.1)
    print(imc.ws_dist_ratio)
    print('diam, vol, eta, beta, N= ', 
          imc.diam, imc.vol, imc.eta, imc.beta, imc.N)
    
    slice_length = int(imc.N_matrix/32)
    Q = itertools.islice(imc.getQ, 32, 32+slice_length)
    tic = time.time()
    
    """
    c = imc.iter_transition()
    next(c)
    next(c)
    print('count=', next(c))
    """
    print('calculating...')
    try:
        q = next(Q)
        q = next(Q)
        #q = next(imc.getQ)
        q = next(imc.Q)
        #q = next(imc.Q)
        print(q)
        #print('count=', imc.get_transition_grid(q))
        row = imc.getrow_ref(q)
        row2 = imc.getrow(q)
        row3 = list(imc.getrow(q))
        for i in range(0, row3[-1]):
            print(str(row3[i][0]) + " " + str(row3[i][1])+ " " + str(row3[i][2]))
        """
        for j in range(3000):
            #print(*next(row2))
            print(next(row2))
            #print('intvl=', next(row))
        """
    except StopIteration:
            print('Stopped at=', 0)
    finally:
        print('Computation time of a single row is {} sec'.format(time.time()-tic))
    




















