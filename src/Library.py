#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:45:14 2022

@author: ym
"""

import math
import itertools
import time
import traceback as tb
import numpy as np
#from multipledispatch import dispatch

#Markoc Set Chain class
#Inputs: P (a (possibly non-stochastic) matrix of lower bounds)
#and Q (a (possibly non-stochastic) matrix of upper bounds)
#Return: An MSC, i.e., every row (i) of MSC is a convex polytope of 
#probability measures (a subset of a hyperplane) bounded by P[i] and Q[i].
#Since each row of the final MSC is a convex polytope, it is enough to genearte
#the vertices.
class MSC:
    round_decimal = int(10)
    
    def __init__(self, P, Q):
        try:
            self.dim = len(P)
            self.P = P
            self.Q = Q
            self.vertices = [[ ] for i in range (0, self.dim)]     
            self.vertices_sorted = [[ ] for i in range (0, self.dim)] 
            raise Exception           
        except Exception:
            if not all([len(i) == len(P) for i in P]) or \
               not all([len(i) == len(Q) for i in Q]) or \
               not (len(P) == len(Q)):                                                                                         
                print('Dimension Errors')
                tb.print_exc()
            elif not all(0 <= P[i][j] <= Q[i][j] <= 1 for i in range(0, len(P)) 
                       for j in range(0,len(P))):
                print('Invalid Inputs')  
                tb.print_exc()

        self.__tighten()
        self.__vertices()
        self.__vertices_sort()
    
    def __del__(self):
        print('MSC Class Destructed')
    
    def __repr__(self):
        str1 = 'The stochastic intervals are tight.'
        str2 = 'The dimension of matrices within this Markov set chain is '
        str3 = 'The stochastic intervals are tight after modification.'
        if self.tightness:
            return str1 +'\n' + str2 + '{0.dim}x{0.dim}.\n'.format(self)
        elif self.__tightness__:
            return str3 +'\n' + str2 + '{0.dim}x{0.dim}.\n'.format(self)
        else:
            return 'Errors occur!\n'
    
    @classmethod
    def chdec(cls, dec):
        cls.round_decimal = dec
    
    
    #Every row of P and Q generates a convex set of probability measures. 
    #The property tightness is to check 
    #if there is no free component in deciding the convex region within the
    #space of (discrete) probability measures,
    #i.e., every entry of each row of P and Q should be a tight constraint.
    @property
    def tightness(self):
        p_tight = all(round(self.P[i][j] + math.fsum(self.Q[i]) - self.Q[i][j],
                             self.round_decimal) >= 1 
                    for i in range(0,self.dim) for j in range(0, self.dim))
        q_tight = all(round(self.Q[i][j] + math.fsum(self.P[i]) - self.P[i][j],
                             self.round_decimal) <= 1 
                    for i in range(0,self.dim) for j in range(0, self.dim))
        return (p_tight and q_tight)
    
    
    #Double check if the refined P.tight and Q.tight are tight matrices.
    @property
    def __tightness__(self):
        p_tight = all(round(self.P_tight[i][j] + math.fsum(self.Q_tight[i]) \
                      - self.Q_tight[i][j], self.round_decimal) >= 1 
                    for i in range(0, self.dim) for j in range(0, self.dim))
        q_tight = all(round(self.Q_tight[i][j] + math.fsum(self.P_tight[i]) \
                      - self.P_tight[i][j], self.round_decimal) <= 1 
                    for i in range(0, self.dim) for j in range(0, self.dim))
        return (p_tight and q_tight)
    
    
    #Return tight lower and upper bound matrices.
    def __tighten(self):
        if self.tightness == False:
            self.P_tight = [[round(1-math.fsum(self.Q[i]) + self.Q[i][j], 
                                   self.round_decimal)
                             if round(math.fsum(self.Q[i]) - self.Q[i][j]
                                      + self.P[i][j], self.round_decimal) < 1 
                             else self.P[i][j] 
                             for j in range(0, self.dim)]  
                             for i in range(0, self.dim)]
            self.Q_tight = [[round(1-math.fsum(self.P[i]) + self.P[i][j], 
                                   self.round_decimal)
                             if round(math.fsum(self.P[i]) - self.P[i][j]
                                      + self.Q[i][j], self.round_decimal) > 1 
                             else self.Q[i][j] 
                             for j in range(0, self.dim)]  
                             for i in range(0, self.dim)]
        else:
            self.P_tight = self.P
            self.Q_tight = self.Q
    
    def __row_vertex_generator(self, row_num, col_num):
        p = self.P_tight[row_num]
        q = self.Q_tight[row_num]
        for i in range(0, self.dim):
            if i != col_num:
                yield [p[i], q[i]]       
    
    
    def row_vertices(self, row_num):
        tic = time.time()
        #print('Computing Vertices of row {}...\n'.format(row_num))
        for col in range(0, self.dim):
            I = itertools.product(*self.__row_vertex_generator(row_num, col))
            for i in I:
                a = round(1 - math.fsum(i), self.round_decimal)
                if self.P_tight[row_num][col] <= a <= \
                    self.Q_tight[row_num][col]:
                        x = list(i)
                        x.insert(col, a)
                        self.vertices[row_num].append(x)
        """
        self.vertices[row_num].sort()
        self.vertices[row_num]=list(self.verties[row_num] 
                                   for self.verties[row_num],
                                   _ in itertools.groupby(
                                       self.verties[row_num]))
        """
        print('Computation of current row accomplished in {} \
              seconds.\n'.format(time.time() - tic))
    def __vertices(self):
        for row in range (0, self.dim):
            self.row_vertices(row)
            
    def __vertices_sort(self):
        for row in range (0, self.dim):
            temp = sorted(self.vertices[row])
            self.vertices_sorted[row] = \
                list(temp for temp,_ in itertools.groupby(temp))

""" End of Class MSC"""


""" Begin of Class Box"""
class Box: 
    def __init__(self, box):           
        self.dim = len(box)
        self.X = box
        self.wid = max(self.X[i][1] - self.X[i][0] for i in range(self.dim))
        self.mid = list((self.X[i][0] + self.X[i][1])/2 
                         for i in range(self.dim))
        #translate the box to be centered at the origin
        self.Xcenter = [[self.X[i][0] - self.mid[i], 
                         self.X[i][1] - self.mid[i]] for i in range(self.dim)]
        
    def __del__(self):
        return None
    
    def __repr__(self):
        string = ''
        for i in range(self.dim):
            if i == 0:
                string += '[{0}, {1}]'.format(self.X[i][0], self.X[i][1])
            else:
                string += ', [{0}, {1}]'.format(self.X[i][0], self.X[i][1])
        return '[' + string + ']'
    
    @property
    def intvl(self):
        if self.dim == 1:
            return self.X[0]
        else:
            return math.nan
    
    def mul(self, b, dim):
        if isinstance(b, Box):
            return [min(self.X[dim][0] * b.X[dim][0], 
                        self.X[dim][0] * b.X[dim][1], 
                        self.X[dim][1] * b.X[dim][0], 
                        self.X[dim][1] * b.X[dim][1]), 
                    max(self.X[dim][0] * b.X[dim][0], 
                        self.X[dim][0] * b.X[dim][1], 
                        self.X[dim][1] * b.X[dim][0], 
                        self.X[dim][1] * b.X[dim][1])]
        elif self.dim == len(b):
            return [min(self.X[dim][0] * b[dim][0], 
                        self.X[dim][0] * b[dim][1], 
                        self.X[dim][1] * b[dim][0], 
                        self.X[dim][1] * b[dim][1]), 
                    max(self.X[dim][0] * b[dim][0], 
                        self.X[dim][0] * b[dim][1], 
                        self.X[dim][1] * b[dim][0], 
                        self.X[dim][1] * b[dim][1])]
        return NotImplemented
    
    # 1/[b] of the dimension 'dim'
    def div(self, b, dim):
        if isinstance(b, Box):
            if b.X[dim][0] == 0 and b.X[dim][1] == 0:
                return math.nan
            elif 0 < b.X[dim][0] or 0 > b.X[dim][1]:
                return [1/b.X[dim][1], 1/b.X[dim][0]]
            elif b.X[dim][0] == 0 and 0 < b.X[dim][1]:
                return [1/b.X[dim][1], math.inf]
            elif b.X[dim][0] < 0 and 0 == b.X[dim][1]:
                return [math.inf, 1/b.X[dim][0]]
            elif b.X[dim][0] < 0 and 0 < b.X[dim][1]:
                return [-math.inf, math.inf]        
        
        return NotImplemented
    
    # 1/[self] of the dimension 'dim'
    def rdiv(self, dim):
        if self.X[dim][0] == 0 and self.X[dim][1] == 0:
            return math.nan
        elif 0 < self.X[dim][0] or 0 > self.X[dim][1]:
            return [1/self.X[dim][1], 1/self.X[dim][0]]
        elif self.X[dim][0] == 0 and 0 < self.X[dim][1]:
            return [1/self.X[dim][1], math.inf]
        elif self.X[dim][0] < 0 and 0 == self.X[dim][1]:
            return [math.inf, 1/self.X[dim][0]]
        elif self.X[dim][0] < 0 and 0 < self.X[dim][1]:
            return [-math.inf, math.inf]        


    def __add__(self, b):
        if isinstance(b, Box):
            try:           
                box = [[self.X[i][0] + b.X[i][0], self.X[i][1] + b.X[i][1]] 
                       for i in range(self.dim)]     
                raise Exception
            except Exception:
                if self.dim != b.dim:
                    print('Dimension Error!')
            return Box(box)
        elif type(b) == float or type(b) == int:
            box = [[self.X[i][0] + b, self.X[i][1] + b] 
                   for i in range(self.dim)]
            return Box(box)
        elif self.dim == len(b):
            if type(b[0]) == list:
                box = [[self.X[i][0] + b[i][0], self.X[i][1] + b[i][1]] 
                       for i in range(self.dim)]
            else:
                box = [[self.X[i][0] + b[i], self.X[i][1] + b[i]] 
                       for i in range(self.dim)]
            return Box(box)
            
        return NotImplemented    
    
    def __sub__(self, b):
        if isinstance(b, Box):
            try:           
                box = [[self.X[i][0] - b.X[i][1], self.X[i][1] - b.X[i][0]] 
                       for i in range(self.dim)]     
                raise Exception
            except Exception:
                if self.dim != b.dim:
                    print('Dimension Error!')
            return Box(box)
        elif type(b) == float or type(b) == int:
            box = [[self.X[i][0] - b, self.X[i][1] - b] 
                   for i in range(self.dim)]
            return Box(box)
        elif self.dim == len(b):
            if type(b[0]) == list:
                box = [[self.X[i][0] - b[i][0], self.X[i][1] - b[i][1]] 
                       for i in range(self.dim)]
            else:
                box = [[self.X[i][0] - b[i], self.X[i][1] - b[i]] 
                       for i in range(self.dim)]
            return Box(box)
        
        return NotImplemented
    
    
    def __mul__(self, b):
        if isinstance(b, Box):
            try:           
                box = np.array([self.mul(b, i) 
                                for i in range(self.dim)]).sum(axis = 0)
                raise Exception
            except Exception:
                if self.dim != b.dim:
                    print('Dimension Error!')
            return Box([list(box)])
        elif type(b) == float or type(b) == int:
            box = [[self.X[i][0] * b, self.X[i][1] * b] if b >= 0 
                   else [self.X[i][1] * b, self.X[i][0] * b]
                   for i in range(self.dim)]
            return Box(box)
        elif self.dim == len(b):
            if type(b[0]) == list:
                box = np.array([self.mul(b, i) for i \
                                in range(self.dim)]).sum(axis = 0).tolist()
            else:        
                box = np.array([[min(self.X[i][0] * b[i], self.X[i][1] * b[i])
                                 , max(self.X[i][0] * b[i], 
                                       self.X[i][1] * b[i])] for i \
                                in range(self.dim)]).sum(axis = 0).tolist()
            return Box([box])
        
        return NotImplemented
    
    def __truediv__(self, b):
        if isinstance(b, Box):
            try: 
                box = Box([self.div(b, i) for i in range(self.dim)])
                raise Exception
            except Exception:
                if self.dim != b.dim:
                    print('Dimension Error!')
            return self * box
        elif type(b) == float or type(b) == int:
            try:
                return self * (1/b)
            except ZeroDivisionError:
                return None
        return NotImplemented
    
    def __rtruediv__(self, b):
        box = Box([self.rdiv(i) for i in range(self.dim)])
        return b * box
    
    def __neg__(self):
        return Box([[-self.X[i][1], -self.X[i][0]] for i in range(self.dim)])

    def __rsub__(self, b):
        return -(self - b)
    
    __radd__ = __add__
    __rmul__ = __mul__

       
""" End of Class Box"""



""" Begin of fundamental functions, overloaded as mappings from Box to Box."""
def sqr(b):
    if isinstance(b, Box) and b.dim == 1:
        box = [[0, max((b.X[i][0])**2, (b.X[i][1])**2)] 
               if (b.X[i][0] <= 0 and 0 <= b.X[i][1]) 
               else [min((b.X[i][0])**2, (b.X[i][1])**2), 
                     max((b.X[i][0])**2, (b.X[i][1])**2)] 
               for i in range(b.dim)]
        return Box(box)
    elif type(b) == float or type(b) == int:
        return b**2
    return NotImplemented

#@dispatch(Box, int)
def power(b, n):
    p = n if n >= 0 else -n
    if isinstance(b, Box) and b.dim == 1: 
        if p % 2 == 0:
            box = [[0, max(math.pow(b.X[i][0], p), math.pow(b.X[i][1], p))] 
                   if (b.X[i][0] <= 0 and 0 <= b.X[i][1]) 
                   else [min(math.pow(b.X[i][0], p), math.pow(b.X[i][1], p)), 
                         max(math.pow(b.X[i][0], p), math.pow(b.X[i][1], p))] 
                   for i in range(b.dim)]
            
        else: 
            box = [[math.pow(b.intvl[0], p), math.pow(b.intvl[1], p)]]
        return Box(box) if n >= 0 else 1/Box(box)
    elif type(b) == float or type(b) == int:
        return math.pow(b, n)
    return NotImplemented   

"""
#Overload with non-integer power
#Temporarily not provided
@dispatch(Box, float)
def power(b, n):
    pass

@dispatch(Box, Box)
def power(b, n):
    pass
"""

def sqrt(b):
    if isinstance(b, Box) and b.dim == 1:
        if 0 <= b.X[0][0]:
            box = [math.sqrt(b.X[0][0]), math.sqrt(b.X[0][1])]
            return Box([box])
        elif b.X[0][0] <= 0 and 0 <= b.X[0][1]:
            box = [0, math.sqrt(b.X[0][1])]
            return Box([box])
        elif b.X[0][1] < 0:
            return math.nan
    elif type(b) == float or type(b) == int:
        return math.sqrt(b)
    return NotImplemented

def exp(b):
    if isinstance(b, Box) and b.dim == 1:
        return Box([[math.exp(b.X[0][0]), math.exp(b.X[0][1])]])
    elif type(b) == float or type(b) == int:
        return math.exp(b)
    return NotImplemented   

def log(b):
    if isinstance(b, Box) and b.dim == 1:
        if b.intvl[0] <= 0:
            return math.nan
        return Box([[math.log(b.intvl[0]), math.log(b.intvl[1])]])
    elif type(b) == float or type(b) == int:
        return math.log(b)
    return NotImplemented 

def sin(b):
    if isinstance(b, Box) and b.dim == 1:       
        if math.floor(((b + math.pi/2)/(2 * math.pi)).intvl[1]) \
            - math.floor(((b + math.pi/2)/(2 * math.pi)).intvl[0]) >= 1 \
                or type(((b + math.pi/2)/(2 * math.pi)).intvl[0]) == int:
                r_low = -1
        else: 
            r_low = min(math.sin(b.intvl[0]), math.sin(b.intvl[1]))

        
        if math.floor(((b - math.pi/2)/(2 * math.pi)).intvl[1]) \
            - math.floor(((b - math.pi/2)/(2 * math.pi)).intvl[0]) >= 1 \
                or type(((b - math.pi/2)/(2 * math.pi)).intvl[0]) == int:
                r_up = 1
        else:
            r_up = max(math.sin(b.intvl[0]), math.sin(b.intvl[1]))

        return Box([[r_low, r_up]])
    elif type(b) == float or type(b) == int:
        return math.sin(b)
    return NotImplemented   

def cos(b):
    return sin(math.pi/2 - b)
        

def tan(b):
    pass

""" End of fundamental functions."""




""" Begin of inclusion functions."""
#Natural inclusion function: mapping from Box to Box.
#Input the domain and the original function f.
def fn(box, f):    
    if isinstance(box, Box):
        if len(box) != 1:
            b=box
            dim_box_list = [Box([b.X[i]]) for i in range(b.dim)]
            result = [f(dim_box_list)[i].intvl 
                      for i in range(len(f(dim_box_list)))]
            return Box(result)
        else:
            return f(box)
    else:
        if len(box) != 1:
            dim_box_list = [Box([box[i]]) for i in range(len(box))]
            result = [f(dim_box_list)[i].intvl 
                      if isinstance(f(dim_box_list)[i], Box) 
                      else [f(dim_box_list)[i]]*2 
                      for i in range(len(f(dim_box_list)))]
            return Box(result)
        else:
            b = Box(box)
            y = f(b)
            return y


#centered includsion function
def fc(box, f, Lip_const, f_prime=None):
    if isinstance(box, Box):
        #temporarily not needed
        pass
    
    else: 
        if f_prime != None:
            if len(box) == 1:
                return fn(box,f_prime) * Box(box).Xcenter \
                       + f(Box(box).mid[0])
            else:
                result = [(fn(box, f_prime[i]) * Box(box).Xcenter + \
                           f(Box(box).mid)[i]).intvl for i in range(len(box))]
                return Box(result)
        else:
            if len(box) == 1:
                return Lip_const * Box(Box(box).Xcenter) + f(Box(box).mid[0])
            else:
                #need to check type(Box(Box(box).Xcenter)) and 
                #type(([Lip_const] * len(box)) * Box(Box(box).Xcenter)))
                result = [ (([Lip_const] * len(box)) * Box(Box(box).Xcenter) + 
                           f(Box(box).mid)[i]).intvl for i in range(len(box))]
                return Box(result)
            
""" End of inclusion functions."""        

    

if __name__ == '__main__':  
    X = [[-2, 2], [-0.1, 0.1]]



    f = lambda x: [x[0]*x[1]-power(x[0], 3), x[1]]
    fp1 = lambda x: [x[1]-3*sqr(x[0]), x[0]]
    fp2 = lambda x: [0, 1]
    fp = [fp1, fp2]
    

    y = fn(X, f)
    z = fc(X, f, 1, fp)
    print(y)

    print(z)
    






