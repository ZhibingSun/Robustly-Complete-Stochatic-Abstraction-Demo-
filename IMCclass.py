#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:11:53 2022

@author: ym
"""

import Library as lib
import math
import itertools
import time
import numpy as np
from scipy.stats import multivariate_normal



#IMC (abstraction) class passes in 
#state space X as boxes, e.g. X = [[lb1, up1], [lb2, up2], ...];
#precision (completeness parameter) eps;
#f in the dynamics X_{t+1} = f(X_t) + b(X_t)w_t;
#b in the dynamics X_{t+1} = f(X_t) + b(X_t)w_t;

#We use the notations in the paper
#'Robustly Complete Finite-State Abstractions 
#for Verification of Stochastic Systems', FORMATS 2022.

class IMC():
    #The coefficient used to generate a discretization of the state space, 
    #such that the image of the inclusion functions can be bounded by 
    #the prescibed precision.
    kappa_coefficient = 0.50    
    
    #The sup Lipschitz constant of test functions of ws norm.
    eps_margin = 0.999
    
    #Specified ws distance (ratio) w.r.t. the reference measure.
    ball_coefficient = 1 
    
    #Specify an error
    ws_dist_ratio = 0.25
    
    #error tolerance
    err = 0.00001
    
    
    def __init__(self, state, precision, f, b, L_f, L_b, cor, \
                 fp=None, bp=None):
        self.dim = len(state)
        self.X = state
        self.diam = max(self.X[i][1]-self.X[i][0] for i in range(0, self.dim))
        self.vol = math.prod(self.X[i][1]-self.X[i][0] \
                             for i in range(0, self.dim))
        self.precision = precision
        self.eps = self.precision * (self.eps_margin - self.kappa_coefficient)
        self.f = f
        self.b = b
        self.L_f = L_f
        self.L_b = L_b
        self.L_b_max = np.array([L_b]).max()
        self.cor = cor
        self.fp = fp
        self.bp = bp        
        self.kappa = self.kappa_coefficient * self.precision
        self.eta, self.N = self.__get_eta_N()
        self.N_matrix = math.prod(self.N) + 1
        self.w, self.N_incl = self.__get_w_N_grid()
        self.Q = self.getQ
        #__getQ() returns an iterable 
        #__getQ() works faster than using np.mgrid + np.dstack for high dim
        #Q iterates the 'floor' points of ''state'' grids
        self.beta = self.ws_dist_ratio * self.eta if self.L_b == 0 \
            else self.ws_dist_ratio * self.eta / math.sqrt(2)
        #get the size beta of grids for mean and covariance
        
    @staticmethod
    #Evaluate a 1-dim Gaussian distribution from -infty to x 
    #using the antiderivatie
    def Phi(x, mean, sig):
        return 0.5*(math.erf((x - mean) / (sig * math.sqrt(2))))+0.5
    
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
        cls.ws_dist = new_coefficient 
        
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
        #tv_coeff = 2 * (2 + math.sqrt(2*self.dim))      
        eta = self.eps / (self.ws_dist_ratio + 2 * self.ball_coefficient)
        #eta = self.eps / (tv_coeff + 1)  
        N = self.__adjust_N(eta)
        eta_refine = (self.vol / math.prod(self.__adjust_N(eta))) \
            ** (1 / self.dim)
        return eta_refine, list(N)
    
    #generate the discretized grids
    @property
    def getQ(self):
        def Qgenerator(self):
            for i in range(0, self.dim):
                yield np.linspace(self.X[i][0], self.X[i][1], \
                                  self.N[i]).tolist()
        return itertools.product(*Qgenerator(self))   
    
    
    #Get the size of [y], i.e. w([y])
    #for inclusion functions for each grid
    def __get_w_N_grid(self):
        vol = math.prod(self.eta \
                             for i in range(0, self.dim))
        w = math.sqrt((self.ball_coefficient * self.kappa)**2 \
                            / (self.L_f**2 + self.L_b_max**2))
        w_input = self.eta + 0.000001 if w >= self.eta else w
        N = self.__adjust_N_grid(w_input)
        w_refine = (vol / math.prod(self.__adjust_N_grid(w))) ** (1 / self.dim)
        print('w_refine=', w_refine)
        print('N=', list(self.__adjust_N_grid(w_input)))
        return w_refine, list(N)
    
    #generate [y] 
    def __get_incl_meas_grid(self, grid):
        def Qgenerator(self):
            for i in range(0, self.dim):
                yield np.linspace(grid[i][0], grid[i][1], 
                                  self.N_incl[i]).tolist()
        return itertools.product(*Qgenerator(self))  
    
    #generate [mean]       
    def __overapprx_mean_row(self, grid_point):
        grid = [[grid_point[i], grid_point[i] + self.eta] \
                for i in range(self.dim)]
        Q_meas = self.__get_incl_meas_grid(grid)
        for q in Q_meas:
            g_box = [[q[i], q[i] + self.w] for i in range(self.dim)]
            yield lib.fc(g_box, self.f, 2, self.fp)
            
    #generate [var]       
    def __overapprx_var_row(self, grid_point):
        grid = [[grid_point[i], grid_point[i] + self.eta] \
                for i in range(self.dim)]
        Q_meas = self.__get_incl_meas_grid(grid)
        for q in Q_meas:
            g_box = [[q[i], q[i] + self.w] for i in range(self.dim)]
            result = [lib.fc([g_box[i]], self.b[i][j], self.L_b[i][j]) \
                      if self.L_b[i][j] != 0 else self.b[i][j] \
                          for i in range(self.dim) for j in range(self.dim)]
            yield result
    
    #discretize [mean] by beta and find ref points of mean
    def __mean_ref(self, grid_point):
        #Generate ref point of mean.
        def mean_ref_generator(self):
            for mean_box in self.__overapprx_mean_row(grid_point):
                for i in range(self.dim):
                    a = math.floor(mean_box.X[i][0] / self.beta) * self.beta
                    yield np.mgrid[a:mean_box.X[i][1]:self.beta] 
        return itertools.product(*mean_ref_generator(self))  
    
    #discretize [var] by beta and find ref points of mean
    def __var_ref(self, grid_point):
        #Generate ref point of mean.
        def var_ref_generator(self):
            for var_box in self.__overapprx_var_row(grid_point):
                for var in var_box:
                    if isinstance(var, lib.Box):
                        a = math.floor(var.X[0][0] / self.beta) * self.beta
                        yield np.mgrid[a:var.X[0][1]:self.beta] 
                    else:
                        yield [math.floor(var / self.beta) * self.beta]
        return itertools.product(*var_ref_generator(self))  
    
    #normalize arr and generate the entries of IMC
    def __prob_intvl(self, arr):
        if abs(max(arr)) < self.err: 
            return [0, 0]
        elif min(arr) - self.beta <= 0 and max(arr) + self.beta <= 1:
            return [0, max(arr) + self.beta]
        elif min(arr) - self.beta > 0 and max(arr) + self.beta <= 1:
            return [min(arr) - self.beta, max(arr) + self.beta]
        elif min(arr) - self.beta <= 0 and max(arr) + self.beta > 1:
            return [0, 1]
        else:
            return [min(arr) - self.beta, 1]

    def __getrow_q(self, grid_point, q):
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
        arr = []
        if self.L_b == 0: 
            for mean in self.__mean_ref(grid_point):
                if self.dim == 2 and not self.cor:
                    arr.append((self.Phi(q[0] + self.eta, mean[0],            \
                                                               self.b[0][0])  \
                                - self.Phi(q[0], mean[0], self.b[0][0]))  *   \
                               (self.Phi(q[1] + self.eta, mean[1],            \
                                                                self.b[1][1]) \
                                - self.Phi(q[1], mean[1], self.b[1][1])))
            
                elif self.dim != 2 and not self.cor:            
                    arr.append( math.prod((self.Phi(q[i]                      \
                                    + self.eta, mean[i], self.b[i][i])        \
                                    - self.Phi(q[i], mean[i], self.b[i][i]))  \
                                                  for i in range(self.dim))) 
                        
                elif self.dim == 2 and self.cor:
                    rv = multivariate_normal(mean, self.b)
                    arr.append(rv.cdf((q[0] + self.eta, q[1] + self.eta))     \
                      + rv.cdf((q[0], q[1])) -rv.cdf((q[0], q[1] + self.eta)) \
                      - rv.cdf((q[0] + self.eta, q[1])))
                        
                else:
                    rv = multivariate_normal(mean, self.b)
                    I = itertools.product(*((1, -1) for i in range(self.dim)))
                    J = itertools.product(*((q[i], q[i] + self.eta)           \
                                                  for i in range(self.dim)))
                    def ge():
                        for i, j in zip(I, J):
                            yield rv.cdf(i) * math.prod(i)                    
                    arr.append(sum(ge()))                    
        else: 
            for mean in self.__mean_ref(grid_point):
                for var_ref in self.__var_ref(grid_point):
                    var = np.array(var_ref).reshape(np.array(self.b).shape)
                    if self.dim == 2 and not self.cor:
                        arr.append((self.Phi(q[0] + self.eta, mean[0],        \
                                                                   var[0][0]) \
                                    - self.Phi(q[0], mean[0], var[0][0])) \
                             * (self.Phi(q[1] + self.eta, mean[1], var[1][1]) \
                                    - self.Phi(q[1], mean[1], var[1][1])))
                    elif self.dim != 2 and not self.cor:            
                        arr.append( math.prod((self.Phi(q[i]                  \
                                        + self.eta, mean[i], var[i][i])       \
                                        - self.Phi(q[i], mean[i], var[i][i])) \
                                                   for i in range(self.dim)))
                    elif self.dim == 2 and self.cor:
                        rv = multivariate_normal(mean, var)
                        arr.append(rv.cdf((q[0] + self.eta, q[1] + self.eta)) \
                                                       + rv.cdf((q[0], q[1])) \
                                             -rv.cdf((q[0], q[1] + self.eta)) \
                                             -rv.cdf((q[0] + self.eta, q[1])))
                    else:
                        rv = multivariate_normal(mean, var)
                        I = itertools.product(*((1, -1) \
                                                for i in range(self.dim)))
                        J = itertools.product(*((q[i], q[i] + self.eta) \
                                                for i in range(self.dim)))
                        def ge():
                            for i, j in zip(I, J):
                                yield rv.cdf(i) * math.prod(i)                    
                        arr.append(sum(ge()))          
                   
        return self.__prob_intvl(arr)
    
    
    def __getrow_cemetary_state(self, grid_point):
        #Evaluate the transition probability T(grid_point, Delta).
        #We generate the refs of transition measure T(grid_point) (Gaussians)
        #The evaluation remains the same as __getrow_q 
        arr = []
        if self.L_b == 0: 
            for mean in self.__mean_ref(grid_point):
                if self.dim == 2 and not self.cor:
                    arr.append(1 - (self.Phi(self.X[0][1], mean[0],           \
                                                                self.b[0][0]) \
                             - self.Phi(self.X[0][0], mean[0], self.b[0][0])) \
                             * (self.Phi(self.X[1][1], mean[1], self.b[1][1]) \
                             - self.Phi(self.X[1][0], mean[1], self.b[1][1])))
            
                elif self.dim != 2 and not self.cor:            
                    arr.append(1 - math.prod((self.Phi(self.X[i][1],          \
                                                       mean[i], self.b[i][i]) \
                             - self.Phi(self.X[i][0], mean[i], self.b[i][i])) \
                                                   for i in range(self.dim))) 
                        
                elif self.dim == 2 and self.cor:
                    rv = multivariate_normal(mean, self.b)
                    rv_cdf = rv.cdf((self.X[0][1], self.X[1][1]))             \
                             + rv.cdf((self.X[0][0], self.X[1][0]))           \
                             - rv.cdf((self.X[0][0], self.X[1][1]))           \
                             - rv.cdf((self.X[0][1], self.X[1][0]))
                    arr.append(1 - rv_cdf)
                        
                else:
                    rv = multivariate_normal(mean, self.b)
                    I = itertools.product(*((1, -1) for i in range(self.dim)))
                    J = itertools.product(*((self.X[i][0], self.X[i][1])      \
                                                   for i in range(self.dim)))
                    def ge():
                        for i, j in zip(I, J):
                            yield rv.cdf(i) * math.prod(i)                    
                    arr.append(1 - sum(ge()))                    
        else: 
            for mean in self.__mean_ref(grid_point):
                for var_ref in self.__var_ref(grid_point):
                    var = np.array(var_ref).reshape(np.array(self.b).shape)
                    if self.dim == 2 and not self.cor:
                        arr.append(1 - (self.Phi(self.X[0][1], mean[0],       \
                                                                   var[0][0]) \
                                - self.Phi(self.X[0][0], mean[0], var[0][0])) \
                                * (self.Phi(self.X[1][1], mean[1], var[1][1]) \
                                - self.Phi(self.X[1][0], mean[1], var[1][1])))
                    elif self.dim != 2 and not self.cor:            
                        arr.append(1 - math.prod((self.Phi(self.X[i][1],      \
                                                          mean[i], var[i][i]) \
                                    - self.Phi(self.X[i][0],                  \
                                                         mean[i], var[i][i])) \
                                                    for i in range(self.dim))) 
                            
                    elif self.dim == 2 and self.cor:
                        rv = multivariate_normal(mean, var)
                        rv_cdf = rv.cdf((self.X[0][1], self.X[1][1]))         \
                                 + rv.cdf((self.X[0][0], self.X[1][0]))       \
                                 - rv.cdf((self.X[0][0], self.X[1][1]))       \
                                 - rv.cdf((self.X[0][1], self.X[1][0]))
                        arr.append(1 - rv_cdf)
                            
                    else:
                        rv = multivariate_normal(mean, var)
                        I = itertools.product(*((1, -1) \
                                                for i in range(self.dim)))
                        J = itertools.product(*((self.X[i][0], self.X[i][1]) \
                                                for i in range(self.dim)))
                        def ge():
                            for i, j in zip(I, J):
                                yield rv.cdf(i) * math.prod(i)                    
                        arr.append(1 - sum(ge()))     
                   
        return self.__prob_intvl(arr)
    
    #get a row of IMC abstraction
    def getrow(self, grid_point):
        for q in self.getQ:
            yield self.__getrow_q(grid_point, q)
        
        yield self.__getrow_cemetary_state(grid_point)
        
    
    def __getrow_q_ref(self, grid_point, q):
        mean = next(self.__mean_ref(grid_point))
        var_ref = next(self.__var_ref(grid_point))
        var_reshape = np.array(var_ref).reshape(np.array(self.b).shape)
        var = self.b if self.L_b == 0 else var_reshape
        if self.dim == 2 and not self.cor:
            prob = (self.Phi(q[0] + self.eta, mean[0], var[0][0])             \
                                    - self.Phi(q[0], mean[0], var[0][0]))     \
                     * (self.Phi(q[1] + self.eta, mean[1],  var[1][1])        \
                                    - self.Phi(q[1], mean[1], var[1][1]))
        elif self.dim != 2 and not self.cor:   
            prob = 1 - math.prod((self.Phi(self.X[i][1], mean[i], var[i][i])  \
                                - self.Phi(self.X[i][0], mean[i], var[i][i])) \
                                                    for i in range(self.dim))
        elif self.dim == 2 and self.cor:
            rv = multivariate_normal(mean, var)
            prob = rv.cdf((q[0] + self.eta, q[1] + self.eta))                 \
                 + rv.cdf((q[0], q[1])) -rv.cdf((q[0], q[1] + self.eta))      \
                 - rv.cdf((q[0] + self.eta, q[1]))
        else:
            rv = multivariate_normal(mean, var)
            I = itertools.product(*((1, -1) for i in range(self.dim)))
            J = itertools.product(*((q[i], q[i] + self.eta)           \
                                              for i in range(self.dim)))
            def ge():
                for i, j in zip(I, J):
                    yield rv.cdf(i) * math.prod(i)                    
            prob = sum(ge())      
        return prob
    
    def __getrow_cemetary_state_ref(self, grid_point):
        mean = next(self.__mean_ref(grid_point))
        var_ref = next(self.__var_ref(grid_point))
        var_reshape = np.array(var_ref).reshape(np.array(self.b).shape)
        var = self.b if self.L_b == 0 else var_reshape
        if self.dim == 2 and not self.cor:
            prob = 1 - (self.Phi(self.X[0][1], mean[0], var[0][0])            \
                      - self.Phi(self.X[0][0], mean[0], var[0][0]))           \
                     * (self.Phi(self.X[1][1], mean[1], var[1][1])            \
                      - self.Phi(self.X[1][0], mean[1], var[1][1]))
        elif self.dim != 2 and not self.cor:            
            prob = 1 - math.prod((self.Phi(self.X[i][1], mean[i], var[i][i])  \
                        - self.Phi(self.X[i][0], mean[i], var[i][i]))         \
                                            for i in range(self.dim))
                
        elif self.dim == 2 and self.cor:
            rv = multivariate_normal(mean, var)
            rv_cdf = rv.cdf((self.X[0][1], self.X[1][1]))         \
                     + rv.cdf((self.X[0][0], self.X[1][0]))       \
                     - rv.cdf((self.X[0][0], self.X[1][1]))       \
                     - rv.cdf((self.X[0][1], self.X[1][0]))
            prob = 1 - rv_cdf
                
        else:
            rv = multivariate_normal(mean, var)
            I = itertools.product(*((1, -1) \
                                    for i in range(self.dim)))
            J = itertools.product(*((self.X[i][0], self.X[i][1]) \
                                    for i in range(self.dim)))
            def ge():
                for i, j in zip(I, J):
                    yield rv.cdf(i) * math.prod(i)                    
            prob = 1 - sum(ge())     
                
        return prob 

            
    
    def getrow_ref(self, grid_point):
        for q in self.getQ:
            yield self.__getrow_q_ref(grid_point, q)    
        yield self.__getrow_cemetary_state_ref(grid_point)


    





if __name__ == '__main__':
    X = [[-1, 1], [-1, 1]]
    precision = 0.2
    #f = lambda x: x
    #b = lambda x: 0.01
    L_f = 2/5
    L_b = [[0.05, 0], [0, 0.05]] 
    cor = False

    f = lambda x: [0.2*lib.sqr(x[1]) - 0.2*x[0], -0.2*x[1]]
    fp1 = lambda x: [-1/5, 2*x[1]/5]
    fp2 = lambda x: [0, -1/5]
    fp = [fp1, fp2]
    b = [[lambda x: 0.05*x, 0], [0, lambda x: 0.05*x]]

    imc = IMC(X, precision, f, b, L_f, L_b, cor, fp)
    print('diam, vol, eta, beta, N= ', 
          imc.diam, imc.vol, imc.eta, imc.beta, imc.N)
    #print(next(imc.Q))
    #print(next(imc.Q))
    
    tic = time.time()

    print('calculating...')
    try:
        q = next(imc.Q)
        #mean_ref=imc.mean_ref(q)
        row = imc.getrow_ref(q)
        
        for j in range(3000):
        #print('imc_q=', q)
            #overapprx=imc.overapprx_mean_row(q)
            next(row)
            #print('intvl=', next(row))
        
    except StopIteration:
            print('Stopped at=', j)
    finally:
        print(time.time()-tic)

















