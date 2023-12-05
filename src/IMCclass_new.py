#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:11:53 2022
Updated on Sat Apr 08 08:17:51 2023

@author: ym
"""

import src.Library as lib
import math
import itertools
import time
import numpy as np
# from scipy.stats import multivariate_normal
from scipy.stats import norm
# from scipy.special import erf
# from collections import deque

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
    def __init__(self, state, precision, f, b, L_f, L_b, \
                 kappa_coefficient=0.50,eps_margin=0.999,\
                 ball_coefficient=1,ws_dist_ratio = 1.3, err=1e-8,\
                 cor=None, fp=None, bp=None, use_fn_f=None, use_fn_b=None):
        #The coefficient used to generate a pseudo-discretization of the state 
        #space during the generation of the image of inclusion functions, 
        #such that the distance between the image of the inclusion functions 
        #and the actual image of the nonlinear vector field can be bounded by 
        #the prescibed precision times the kappa_coefficient.
        self.kappa_coefficient = kappa_coefficient
        
        #Set a ratio that is within (0, 1) for the prescibed precision
        self.eps_margin = eps_margin
        
        #The sup Lipschitz constant of test functions of ws norm.
        self.ball_coefficient = ball_coefficient
        
        #Specified ws distance ratio w.r.t. the grid size.
        #The multiplication of ws_dist_ratio and the grid size returns 
        #the distance w.r.t. the reference measure.
        self.ws_dist_ratio = ws_dist_ratio  #0.25
        
        #error tolerance
        self.err = err
        
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
        
        eta = self.eps / (self.ws_dist_ratio + 2 * self.ball_coefficient)
        
        #round up the number of grids of the state space
        self.N = [round(math.floor((self.X[i][1] - self.X[i][0]) / eta) + 1)\
             for i in range(0, self.dim)]
        
        #use the complete analysis to obtain the grid size eta (in Eq.(19))
        self.eta = (self.vol / math.prod(self.N)) \
            ** (1 / self.dim)

        self.N_matrix = math.prod(self.N) + 1

        ##begin __get_w_N_grid
        #Get the size of [y], i.e. w([y]),
        #for inclusion functions on each grid.
        #Decide if the size of a state grid, eta,
        #is already small enough to make 
        #the set of Gauss measures within the required Wasserstein distance, 
        #which in this case is self.kappa.
        #Usually, when L_f**2 + L_b_max**2 is large, one needs to further split
        #the grid into [y], and patch up [mean] and [var]; 
        #otherwise we just use the state grid to generate [mean] and [var].
        #
        #Note that when L_f**2 + L_b_max**2 = 0, 
        #[mean] and [var] reduce to singletons.
        vol = math.prod(self.eta for i in range(0, self.dim))
        tmp = self.L_f**2 + self.L_b_max**2
        if tmp != 0:
            w = math.sqrt((self.ball_coefficient * self.kappa)**2 \
                                / tmp)
        else:
            w = self.eta
        w_input = self.eta + 1e-6 if w >= self.eta else w

        #round up the number of grids for discretizing 
        #the domain of inclusion functions
        adjust_N_grid = lambda x: [round(math.floor(self.eta / x) + 1)\
             for i in range(0, self.dim)]

        #a discretization for the generation of inclusion 
        # of the 'reachable set of Gauss measures'.
        self.N_incl = adjust_N_grid(w_input)
        #conservative w
        self.w = (vol / math.prod(adjust_N_grid(w))) ** (1 / self.dim)
        print('w_refine=', self.w)
        print('N=', self.N_incl)
        ##end __get_w_N_grid
        
        #__getQ() returns an iterable 
        #__getQ() works faster than using np.mgrid + np.dstack for high dim
        #Q iterates the 'floor' points of ''state'' grids
        # self.Q = self.getQ
              
        #get the size beta of grids for mean and covariance
        self.beta = self.ws_dist_ratio * self.eta if self.L_b == 0 \
            else self.ws_dist_ratio * self.eta / math.sqrt(2)

        #Generate the discretized grids.
        tmp = [np.linspace(self.X[i][0], self.X[i][1], \
                self.N[i]) for i in range(0, self.dim)]
        self.pt_cube = np.array(list(itertools.product(*tmp))+[[0]*self.dim])
        print(self.pt_cube.shape)
        
    #Evaluate a 1-dim Gaussian distribution from -infty to x 
    #using the antiderivatie
    def Phi(self, x, mean, sig):
        if sig != 0:
            return 0.5*(math.erf((x - mean) / (sig * math.sqrt(2))))+0.5
            # return 0.5*(erf((x - mean) / (sig * math.sqrt(2))))+0.5
        else:
            return int(x > mean)

    # def Phi(self, x, mean, sig):
    #     return norm.cdf(x,mean,sig)
        
    @property
    def dictionary(self):
        return {'dimension_of_workplace': [self.dim],\
               'diameter_of_workplace': [self.diam], \
               'volume_of_workplace': [self.vol], \
               'number_of_grid_workplace': [self.N], \
               'dimension_of_IMC': [self.N_matrix], \
                'gridsize_of_workplace': [self.eta], \
                'gridsize_of_measure': [self.beta]}
    
    # def __get_incl_meas_grid(self, grid):
    #     tmp = [np.linspace(grid[i][0], grid[i][1], \
    #             self.N_incl[i]).tolist() for i in range(0, self.dim)]
    #     return itertools.product(*tmp)
    
    #Generate [mean], generate [std] (or [Cholesky]), which is supposed to be a dim * dim matrix.       
    #In this function, 
    #the dim * dim matrix is reshaped as a dim^2 * 1 list of Boxes.
    def __overapprx_row_mean_std(self, grid_point):
        grid = [[grid_point[i], grid_point[i] + self.eta] \
                for i in range(self.dim)]
        #Generate [y], which is a pseudo-grid to generate the inclusion of the
        #'reachable set of Gauss measures'. 
        #As a continuation of __get_w_N_grid( ), 
        #this function records the 'bottom-left' point of the required-size grids.
        tmp = [np.linspace(grid[i][0], grid[i][1], \
                self.N_incl[i]).tolist() for i in range(0, self.dim)]
        Q_meas = list(itertools.product(*tmp))
        mean = []
        std = []
        for q in Q_meas:
            g_box = [[q[i], q[i] + self.w] for i in range(self.dim)]
            mean.append(lib.fn(g_box, self.f) if self.use_fn_f is not None \
                else (lib.fc(g_box, self.f, self.L_f, self.fp) if self.L_f != 0 \
                      else self.f))
            std.append([lib.fn([g_box[i]], self.b[i][j]) \
                      if self.use_fn_b is not None \
                      else(lib.fc([g_box[i]], self.b[i][j], self.L_b[i][j]) \
                           if self.L_b[i][j] != 0 else self.b[i][j]) \
                          for i in range(self.dim) for j in range(self.dim)])
        return mean, std
    
    #discretize [mean] by beta and find ref points of mean
    #discretize [std] (or [Cholesky]) by beta and find ref points of mean
    def __ref_mean_std(self, grid_point):
        mean_list, std_list = self.__overapprx_row_mean_std(grid_point)
        #Generate ref point of mean.
        tmp1 = []
        for mean_box in mean_list:
            for i in range(self.dim):
                if isinstance(mean_box, lib.Box):
                    a = math.floor(mean_box.X[i][1] / self.beta) * self.beta
                    tmp1.append(np.mgrid[mean_box.X[i][0]:a:self.beta] \
                        if a >= mean_box.X[i][0] else [mean_box.X[i][0]])
                else:
                    tmp1.append([mean_box[i]])
        #Generate ref point of std.
        tmp2 = []
        for std_box in std_list:
            for std in std_box:
                if isinstance(std, lib.Box):
                    a = math.floor(std.X[0][1] / self.beta) * self.beta
                    tmp2.append(np.mgrid[std.X[0][0]:a:self.beta]  \
                        if a >= std.X[0][0] else [std.X[0][0]])
                else:
                    #In this case, the current entry of b is a constant.
                    #The generator should yield an iterable 
                    #rather than a number for the downstream calculation.
                    tmp2.append([std])
                    #yield [math.floor(std / self.beta) * self.beta]
        return list(itertools.product(*tmp1)), list(itertools.product(*tmp2))
    
    def __evaluate_discrete_probability(self, mean, std):
        std_array = np.diag(std)
        # mean_array = np.array(mean)
        # x = (self.pt_cube-mean_array)/std_array
        # y = (self.pt_cube+self.eta-mean_array)/std_array

        # mean_array = np.array([mean]*self.N_matrix)
        # std_array = np.array([np.diag(std).tolist()]*self.N_matrix)

        # tmp1 = norm.cdf((self.eta + self.pt_cube-mean_array)/std_array)
        # tmp2 = norm.cdf((self.pt_cube-mean_array)/std_array)
        tmp_X = np.array(self.X)
        tmp1 = np.zeros((self.N_matrix,self.dim))
        tmp2 = np.zeros((self.N_matrix,self.dim))
        for i in range(self.dim):
            self.pt_cube[-1,i] = tmp_X[i,1]-self.eta
            # tmp1 = norm.cdf(self.eta + self.pt_cube,mean_array,std_array)
            tmp1[:,i] = norm.cdf((self.eta + self.pt_cube[:,i]-mean[i])/std_array[i])
            self.pt_cube[-1,i] = tmp_X[i,0]
            # tmp2 = norm.cdf(self.pt_cube,mean_array,std_array)
            tmp2[:,i] = norm.cdf((self.pt_cube[:,i]-mean[i])/std_array[i])

        # tmp1 = norm.cdf(x)
        # tmp2 = norm.cdf(y)
        result = np.prod(tmp1-tmp2,axis=1)
        result[-1] = 1-result[-1]
        result = np.where(result < self.err, 0, result)
        return result / np.sum(result) #normalize

    # def __evaluate_discrete_probability(self, mean, std, q):
    #     #Evaluate the transition probability T(grid_point, q).
        
    #     #For each q, 
    #     #we generate the refs of transition measure T(grid_point) (Gaussians).
        
    #     #The refs should be duplicated for each q.
        
    #     #For dim == 2, 
    #     #direct multiplication of antiderivative is faster than math.prod.
        
    #     #For L_b != 0, the evaluation depends on the correlation:
    #     #if each dimension is independent, 
    #     #multiplication of antiderivative is 100 times 
    #     #faster than using scipy cdf. 
        
    #     if self.dim == 2 and not self.cor:
    #         return (
    #         return math.prod((self.Phi(q[i] + self.eta, mean[i], std[i][i])   \
    #                             - self.Phi(q[i], mean[i], std[i][i])) self.Phi(q[0] + self.eta, mean[0], std[0][0])             \
    #                         - self.Phi(q[0], mean[0], std[0][0]))             \
    #                  * (self.Phi(q[1] + self.eta, mean[1], std[1][1])         \
    #                         - self.Phi(q[1], mean[1], std[1][1]))
    #     elif self.dim != 2 and not self.cor:               \
    #                                        for i in range(self.dim))
    #     elif self.dim == 2 and self.cor:
    #         var = np.dot(std, std.T)
    #         rv = multivariate_normal(mean, var)
    #         return rv.cdf((q[0] + self.eta, q[1] + self.eta))                 \
    #                                  + rv.cdf((q[0], q[1]))                   \
    #                                  - rv.cdf((q[0], q[1] + self.eta))        \
    #                                  - rv.cdf((q[0] + self.eta, q[1]))
    #     else:
    #         var = np.dot(std, std.T)
    #         rv = multivariate_normal(mean, var)
    #         I = itertools.product(*((-1, 1) for i in range(self.dim)))
    #         J = itertools.product(*((q[i], q[i] + self.eta) \
    #                                     for i in range(self.dim)))
    #         arr = np.array([rv.cdf(j) * math.prod(i) for i, j in zip(I, J)])    
    #         return np.sum(arr)
            
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
        mean_ref_list, std_ref_list = self.__ref_mean_std(grid_point)
        result = []
        for mean in mean_ref_list:
            for std_ref in std_ref_list:
                std = np.array(std_ref).reshape(np.array(self.b).shape)
                std = np.abs(std)
                result.append(self.__evaluate_discrete_probability(mean, std))
        return result
                
    def __get_row_ref_measure(self, grid_point):
        mean = next(self.__mean_ref(grid_point))
        std_ref = next(self.__std_ref(grid_point))
        std_reshape = np.array(std_ref).reshape(np.array(self.b).shape)
        std = self.b if self.L_b == 0 else std_reshape
        row_ref_measure_grid = \
            np.array([self.__evaluate_discrete_probability(mean, std, \
                                            q) for q in self.pt_cube])
        row_ref_measure_cemetary = \
            np.array([self.__evaluate_cemetary_probability(mean, std)])
        row_measure = \
            np.hstack((row_ref_measure_grid, row_ref_measure_cemetary))
        return row_measure / np.sum(row_measure)
    
    #Generate the entries of IMC
    # def __bounds_row_measure(self, grid_point):
    #     row_measure = self.__get_row_measure(grid_point)
    #     upper_bounds = self.beta + \
    #         np.maximum.reduce(row_measure)
    #     lower_bounds = -self.beta + \
    #         np.minimum.reduce(row_measure)
        
    #     #add filters
    #     lower_bounds = np.where(lower_bounds < self.err, 0, lower_bounds)
    #     lower_bounds = \
    #         np.where(lower_bounds > 1 - self.err -self.beta, 1, lower_bounds)
    #     upper_bounds = np.where(upper_bounds > 1, 1, upper_bounds)
    #     upper_bounds = \
    #         np.where(upper_bounds < self.err + self.beta, 0, upper_bounds)
    #     # return list(zip(lower_bounds, upper_bounds))
    #     return lower_bounds, upper_bounds
    
    def __bounds_row_measure(self, grid_point):
        
        
        #add filters
        lower_bounds = np.where(lower_bounds < self.err, 0, lower_bounds)
        lower_bounds = \
            np.where(lower_bounds > 1 - self.err -self.beta, 1, lower_bounds)
        upper_bounds = np.where(upper_bounds > 1, 1, upper_bounds)
        upper_bounds = \
            np.where(upper_bounds < self.err + self.beta, 0, upper_bounds)
        # return list(zip(lower_bounds, upper_bounds))
        return lower_bounds, upper_bounds
    
    def output(self, grid_point):
        row_measure = self.__get_row_measure(grid_point)
        upper_bounds = self.beta + \
            np.maximum.reduce(row_measure)
        upper_bounds = np.where(upper_bounds > 1, 1, upper_bounds)
        upper_bounds = \
            np.where(upper_bounds < self.err + self.beta, 0, upper_bounds)
        lower_bounds = -self.beta + \
            np.minimum.reduce(row_measure)
        lower_bounds = np.where(lower_bounds < self.err, 0, lower_bounds)
        lower_bounds = \
            np.where(lower_bounds > 1 - self.err -self.beta, 1, lower_bounds)
        row = []
        for i, j in enumerate(upper_bounds):
            if j:
                # if j > 1:
                #     j = 1
                # if lower_bounds[i] < self.err:
                #     lower_bounds[i] = 0
                # if lower_bounds[i] > 1 - self.err -self.beta:
                #     lower_bounds[i] = 1
                row.append([i,lower_bounds[i],j])
        return row

    # def output(self, grid_point):
    #     row = []
    #     for i, j in enumerate(self.__bounds_row_measure(grid_point)):
    #         if j[1]:
    #             row.append([i,j[0],j[1]])
    #     return row
            
    def getrow_ref(self, grid_point):
        ref_measure = self.__get_row_ref_measure(grid_point)
        assert ref_measure.shape[0] == self.N_matrix
        nonzero_index = np.nonzero(ref_measure)
        for i in nonzero_index[0]:
            yield i, ref_measure[i]
        
    def getQ_slice(self, i, portion):
        slice_length = int(self.N_matrix / portion) + 1
        j = i * slice_length
        Q_iter = itertools.islice(self.getQ(), j, j + slice_length)
        return j, Q_iter
