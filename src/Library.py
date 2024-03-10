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

""" Begin of Class Box"""
class Box: 
    def __init__(self, box):           
        self.dim = len(box)
        self.X = box
        
    @property
    def wid(self):
        return max(self.X[i][1] - self.X[i][0] for i in range(self.dim))

    @property
    def mid(self):
        return [(self.X[i][0] + self.X[i][1])/2 for i in range(self.dim)]

    #translate the box to be centered at the origin
    @property
    def Xcenter(self):
        return [[self.X[i][0] - self.mid[i], 
                         self.X[i][1] - self.mid[i]] for i in range(self.dim)]

    def __repr__(self):
        string = ''
        for i in range(self.dim):
            if i == 0:
                string += f'[{self.X[i][0]}, {self.X[i][1]}]'
            else:
                string += f', [{self.X[i][0]}, {self.X[i][1]}]'
        return '[' + string + ']'
    
    @property
    def intvl(self):
        if self.dim == 1:
            return self.X[0]
        else:
            return math.nan
    
    def mul(self, b, dim):
        if isinstance(b, Box): # not used
            tmp = [self.X[dim][0] * b.X[dim][0], 
                        self.X[dim][0] * b.X[dim][1], 
                        self.X[dim][1] * b.X[dim][0], 
                        self.X[dim][1] * b.X[dim][1]]
            return [min(*tmp),max(*tmp)]
        elif self.dim == len(b):
            tmp = [self.X[dim][0] * b[dim][0], 
                        self.X[dim][0] * b[dim][1], 
                        self.X[dim][1] * b[dim][0], 
                        self.X[dim][1] * b[dim][1]]
            return [min(*tmp),max(*tmp)]
        return NotImplemented
    
    # 1/[b] of the dimension 'dim'
    def div(self, b, dim):
        if isinstance(b, Box):
            if b.X[dim][0] == 0 and b.X[dim][1] == 0:
                return math.nan # the empty set is expressed as [math.nan, math.nan]
            elif 0 < b.X[dim][0] or 0 > b.X[dim][1]:
                return [1/b.X[dim][1], 1/b.X[dim][0]]
            elif b.X[dim][0] == 0 and 0 < b.X[dim][1]:
                return [1/b.X[dim][1], math.inf]
            elif b.X[dim][0] < 0 and 0 == b.X[dim][1]:
                return [math.inf, 1/b.X[dim][0]] # wrong, [-math.inf, 1/b.X[dim][0]]
            elif b.X[dim][0] < 0 and 0 < b.X[dim][1]:
                return [-math.inf, math.inf]        
        return NotImplemented
    
    # 1/[self] of the dimension 'dim'
    def rdiv(self, dim):
        if self.X[dim][0] == 0 and self.X[dim][1] == 0:
            return math.nan # empty set is expressed as [math.nan, math.nan]
        elif 0 < self.X[dim][0] or 0 > self.X[dim][1]:
            return [1/self.X[dim][1], 1/self.X[dim][0]]
        elif self.X[dim][0] == 0 and 0 < self.X[dim][1]:
            return [1/self.X[dim][1], math.inf]
        elif self.X[dim][0] < 0 and 0 == self.X[dim][1]:
            return [math.inf, 1/self.X[dim][0]] # wrong, [-math.inf, 1/self.X[dim][0]]
        elif self.X[dim][0] < 0 and 0 < self.X[dim][1]:
            return [-math.inf, math.inf]        

    def __add__(self, b):
        if isinstance(b, Box):
            if self.dim != b.dim:
                print("__add__ dimension mismatch.")
                raise Exception
            box = [[self.X[i][0] + b.X[i][0], self.X[i][1] + b.X[i][1]] 
                       for i in range(self.dim)]
            return Box(box)
        elif type(b) == float or type(b) == int:
            box = [[self.X[i][0] + b, self.X[i][1] + b] 
                   for i in range(self.dim)]
            return Box(box)
        elif type(b)==list and self.dim == len(b):
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
            if self.dim != b.dim:
                print("__sub__ dimension mismatch.")
                raise Exception
            box = [[self.X[i][0] - b.X[i][1], self.X[i][1] - b.X[i][0]] 
                       for i in range(self.dim)]     
            return Box(box)
        elif type(b) == float or type(b) == int:
            box = [[self.X[i][0] - b, self.X[i][1] - b] 
                   for i in range(self.dim)]
            return Box(box)
        elif type(b)==list and self.dim == len(b):
            if type(b[0]) == list:
                box = [[self.X[i][0] - b[i][0], self.X[i][1] - b[i][1]] # wrong, [self.X[i][0] - b[i][1], self.X[i][1] - b[i][0]]
                       for i in range(self.dim)]
            else:
                box = [[self.X[i][0] - b[i], self.X[i][1] - b[i]] 
                       for i in range(self.dim)]
            return Box(box)
        return NotImplemented
    
    def __mul__(self, b): # This is actually dot product.
        if isinstance(b, Box): # not used
            if self.dim != b.dim:
                print("__mul__ dimension mismatch.")
                raise Exception
            box = np.array([self.mul(b, i) 
                        for i in range(self.dim)]).sum(axis = 0)
            return Box([list(box)])
        elif type(b) == float or type(b) == int:
            box = [[self.X[i][0] * b, self.X[i][1] * b] if b >= 0 # see 157 row
                   else [self.X[i][1] * b, self.X[i][0] * b]
                   for i in range(self.dim)]
            return Box(box)
        elif type(b)==list and self.dim == len(b):
            if type(b[0]) == list: # dot product for cif with gradient interval
                box = np.array([self.mul(b, i) for i in range(self.dim)]) \
                               .sum(axis = 0).tolist()
            else: # not used, this can be used for cif with gradient to be scalar not interval
                box = np.array([[min(self.X[i][0] * b[i], self.X[i][1] * b[i]),
                                 max(self.X[i][0] * b[i], self.X[i][1] * b[i])] 
                                 for i in range(self.dim)]).sum(axis = 0).tolist()
            return Box([box])
        
        return NotImplemented
    
    def __truediv__(self, b):
        if isinstance(b, Box):
            if self.dim != b.dim:
                print("__truediv__ dimension mismatch.")
                raise Exception
            box = Box([self.div(b, i) for i in range(self.dim)]) # Why need self.div? Use rdiv is enough.
            return self * box # wrong, * now is dot product, not multiply in each dimension.
        elif type(b) == float or type(b) == int:
            return self * (1/b) # Manipulate directly can reduce checks.
            # wrong, * now is dot product, not multiply in each dimension.
        return NotImplemented
    
    def __rtruediv__(self, b):
        box = Box([self.rdiv(i) for i in range(self.dim)])
        return b * box # b should be const.
        # wrong, * now is dot product, not multiply in each dimension.
    
    def __neg__(self):
        return Box([[-self.X[i][1], -self.X[i][0]] for i in range(self.dim)])

    def __rsub__(self, b): # b should be const, so should be equivalent to self.__neg__ + b
        return -(self - b)
    
    __radd__ = __add__
    __rmul__ = __mul__

""" End of Class Box"""

""" Begin of elementary interval functions, overloaded as mappings from Box to Box."""
def sqr(b):
    if isinstance(b, Box) and b.dim == 1: # wrong classification
        box = [[0, max((b.X[i][0])**2, (b.X[i][1])**2)] 
               if (b.X[i][0] <= 0 and b.X[i][1] >= 0)
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
    if isinstance(b, Box) and b.dim == 1:  # wrong classification
        if p % 2 == 0:
            box = [[0, max(math.pow(b.X[i][0], p), math.pow(b.X[i][1], p))] 
                   if (b.X[i][0] <= 0 and 0 <= b.X[i][1]) 
                   else [min(math.pow(b.X[i][0], p), math.pow(b.X[i][1], p)), 
                         max(math.pow(b.X[i][0], p), math.pow(b.X[i][1], p))] 
                   for i in range(b.dim)]
        else: 
            box = [[math.pow(b.intvl[0], p), math.pow(b.intvl[1], p)]] # Why not write as b.X[i][0], b.X[i][1] now?
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
        elif b.X[0][0] <= 0 and 0 <= b.X[0][1]: # 0 <= b.X[0][1]:
            box = [0, math.sqrt(b.X[0][1])]
            return Box([box])
        elif b.X[0][1] < 0: # else:
            return math.nan # the empty set is expressed as [math.nan, math.nan]
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
        if b.intvl[0] <= 0: # Should be treated as sqrt.
            return math.nan
        return Box([[math.log(b.intvl[0]), math.log(b.intvl[1])]])
    elif type(b) == float or type(b) == int:
        return math.log(b)
    return NotImplemented 

def sin(b):
    if isinstance(b, Box) and b.dim == 1:       
        if math.floor(((b + math.pi/2)/(2 * math.pi)).intvl[1]) \
            - math.floor(((b + math.pi/2)/(2 * math.pi)).intvl[0]) >= 1 \
                or type(((b + math.pi/2)/(2 * math.pi)).intvl[0]) == int: # may be wrong
                r_low = -1
        else: 
            r_low = min(math.sin(b.intvl[0]), math.sin(b.intvl[1]))

        
        if math.floor(((b - math.pi/2)/(2 * math.pi)).intvl[1]) \
            - math.floor(((b - math.pi/2)/(2 * math.pi)).intvl[0]) >= 1 \
                or type(((b - math.pi/2)/(2 * math.pi)).intvl[0]) == int: # may be wrong
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

""" End of elementary interval functions."""

""" Begin of inclusion functions."""
#Natural inclusion function: mapping from Box to Box.
#Input the domain and the original function f.
def fn(box, f):    
    if isinstance(box, Box):
        if len(box) != 1: # len should error
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

#centered inclusion function
def fc(box, f, Lip_const, f_prime=None):
    if isinstance(box, Box):
        #temporarily not needed
        pass
    
    else: 
        if f_prime != None: # we don't need Lip_const if f_prime is not None
            if len(box) == 1:
                return fn(box,f_prime) * Box(box).Xcenter \
                       + f(Box(box).mid[0]) # box + scalar
            else:
                # tmp1 = fn(box, f_prime[0]) * Box(box).Xcenter
                # tmp2 = f(Box(box).mid)[0]
                # print(type(tmp2))
                # tmp3 = tmp1 + tmp2
                result = [(fn(box, f_prime[i]) * Box(box).Xcenter + # box + scalar
                           f(Box(box).mid)[i]).intvl for i in range(len(box))]
                return Box(result)
        else:
            if len(box) == 1:
                return Lip_const * Box(Box(box).Xcenter) + f(Box(box).mid[0]) # box + scalar
            else: # not used
                #need to check type(Box(Box(box).Xcenter)) and 
                #type(([Lip_const] * len(box)) * Box(Box(box).Xcenter)))
                result = [ (([Lip_const] * len(box)) * Box(Box(box).Xcenter) + # can't take in different Lip_const for different dimension
                           f(Box(box).mid)[i]).intvl for i in range(len(box))] # box + scalar
                return Box(result)
            
""" End of inclusion functions."""        
