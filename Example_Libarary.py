#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:52:40 2023

@author: ym
"""
import math
import os 
import Library as lib
#import time
#from collections import ChainMap 
#import logging
#import numpy as np



#Define boxes (intervals in Euclidean space)
#[[interval of dim 0], [interval of dim 1], ... ]
box1 = [[-0.01, 0.01], [-0.01, 0.01]]
box2 = [[0.5, 1.5], [1.5, 2.5]]
box3 = [[math.pi*2/3, math.pi*4/3]]
box4 = [[-1, 1]]


#Generate dictionaries of Box objects
dict1 = {'B1': lib.Box(box1), 'B2': lib.Box(box2), 'B3': lib.Box(box3), 
         'B4': lib.Box(box4)}



#Define test functions 
def test_Box_properties(**kwargs):
    dirname = 'Example_Library_Box_properties'
    if not os.path.isdir('./' + dirname):
        os.mkdir(dirname)
    print('Basic properties of Box objects are shown in directory: ./'+ dirname)
    for key, value in kwargs.items():
        filename = 'test_Box_properties-' + key + '.txt'
        with open(os.path.join('./' + dirname, filename),'w') as f:
            l1 = 'The Box object {0} is: {1}.\n'.format(key, value)
            l2 = 'The width of Box {0} is: {1}.\n'.format(key, value.wid)
            l3 = 'The translated box centered at the origin is: {0}.\n'.format\
                (value.Xcenter)
            l4 = '-{0} is: {1}.'.format(key, -value)
            f.writelines([l1, l2, l3, l4])
    return None





    
    
def test_inclusion_functions(*args):
    dirname = 'Example_Library_Inclusion_functions'
    if not os.path.isdir('./' + dirname):
        os.mkdir(dirname)
    print('Arithmatic properties of Box objects are shown in directory: ./'\
          + dirname)
    for count, func in enumerate(args, start=1):
        filename = 'test_inclusion_functions_' + str(count) + '.txt'
        img_no_prime = lib.fc(func['Dom'], func['f'], func['Lip_const'])
        img_with_prime = lib.fc(func['Dom'], func['f'], \
                                func['Lip_const'], func['f_prime'])
        
        f_type = 'function' if func['dimension']==1 else 'vector field'
        str_join = "%s %s"%('The image of', f_type) 
        l1 = str_join + ' #' + str(count) + \
            ' (without providing the derivatives) is: {}\n'.format(img_no_prime)
        l2 = str_join + ' #' + str(count) + \
            ' (with providing the derivatives) is: {}'.format(img_with_prime)
        with open(os.path.join('./' + dirname, filename),'w') as f:
            f.writelines([l1, l2])
    return None




if __name__ == '__main__':
    #test basic properties
    test_Box_properties(**dict1)

    #Provide dynamics (r.h.s. functions or vector fields)
    dyn1 = {'f': lambda x: lib.sqr(x) - x, 'Lip_const': 1, \
            'f_prime': lambda x: 2*x - 1, 'Dom': box4, 'dimension': 1}
    dyn2 = {'f': lambda x: [-x[1] - x[0]*(lib.sqr(x[0])+lib.sqr(x[1])),\
                            x[0]-x[1]*(lib.sqr(x[0])+lib.sqr(x[1]))], \
            'Lip_const': 4, \
            'f_prime': [lambda x:[-3*lib.sqr(x[0])-lib.sqr(x[1]), -1-2*x[0]*x[1]],\
                        lambda x: [1-2*x[0]*x[1], -3*lib.sqr(x[1])-lib.sqr(x[0])]],\
                'Dom': box1, 'dimension': 2}
    dyn3 = {'f': lambda x: [lib.sqr(x[1]) - x[0], -x[1]], 'Lip_const': 2, \
            'f_prime': [lambda x: [-1, 2*x[1]], lambda x: [0, -1]], \
                'Dom': box1, 'dimension': 2}
    
    #test arithmetic properties
    test_inclusion_functions(dyn1, dyn2, dyn3)




