#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:23:18 2023

@author: ym
"""

from IMCclass import IMC
import Library as lib
import numpy as np
import time
import math
import pandas as pd
import os 


#setup
X = [[-1, 1], [-1, 1]]
precision = 0.2
L_f = 2/5
L_b = [[0, 0], [0, 0]] 
cor = False

f = lambda x: [0.2*lib.sqr(x[1]) - 0.2*x[0], -0.2*x[1]]
fp1 = lambda x: [-1/5, 2*x[1]/5]
fp2 = lambda x: [0, -1/5]
fp = [fp1, fp2]
b = [[math.sqrt(0.5), 0], [0, math.sqrt(0.5)]] 

imc = IMC(X, precision, f, b, L_f, L_b, cor, fp)


dirname = 'Example - IMC - 1_results'
if not os.path.isdir('./' + dirname):
    os.mkdir(dirname)

#output basic information
pd.DataFrame(imc.dictionary).to_csv('Example - IMC - 1_results/Dictionary.csv')
pd.DataFrame({'Grid Point': list(imc.getQ)}).\
    to_csv('Example - IMC - 1_results/Grid_points.csv')
print(imc.dictionary)


#output IMC
M = 2 #set a number of rows for output; should be <= imc.N_matrix
tic1 = time.time()
print('calculating IMC abstraction...')
df = pd.DataFrame([list(imc.getrow(q)) for q in list(imc.Q)[:M]], \
                  columns=np.mgrid[0 : imc.N_matrix : 1])                 
print('Computation time of IMC abstraction = {} sec'.format(time.time() - tic1))

tic2 = time.time()
df.to_csv('Example - IMC - 1_results/IMC_abstraction_matrix.csv')
print('Data storage time of IMC abstraction = {} sec\n'.format(time.time() - tic2))


#output reference stochastic matrix
M = 3 #set a number of rows for output; should be <= imc.N_matrix
tic3 = time.time()
print('calculating reference stochastic matrix...')
df2 = pd.DataFrame([list(imc.getrow_ref(q)) for q in list(imc.Q)[:M]], \
                  columns=np.mgrid[0 : imc.N_matrix : 1])                 
print('Computation time of ref stochastic matrix = {} sec'.format(time.time() - tic3))

tic4 = time.time()
df2.to_csv('Example - IMC - 1_results/Reference_stochastic_matrix.csv')
print('Data storage time ref stochastic matrix = {} sec'.format(time.time() - tic4))

