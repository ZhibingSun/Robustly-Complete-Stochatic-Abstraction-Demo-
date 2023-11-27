#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 01 10:19:52 2023

@author: ym
"""

from IMCclass_new import IMC
import Library as lib
import numpy as np
import time
import math
import pandas as pd
import os 


#Set up for a two-dim Brownian Motion with dt = 0.001**2
X = [[-1, 1], [-1, 1]]
precision = 0.08
L_f = 1
L_b = [[0, 0], [0, 0]] 
cor = False


f = lambda x: [x[0], x[1]]
fp1 = lambda x: [1, 0]
fp2 = lambda x: [0, 1]
fp = [fp1, fp2]
b = [[0.001, 0], [0, 0.001]] 

imc = IMC(X, precision, f, b, L_f, L_b, cor, fp)
#imc.chws_dist(0.001)

"""
dirname = 'Example_IMC_2-results'
if not os.path.isdir('./' + dirname):
    os.mkdir(dirname)
"""
#output basic information
pd.DataFrame(imc.dictionary).to_csv('Example_IMC_2-results/Dictionary.csv')
pd.DataFrame({'Grid Point': list(imc.getQ)}).\
    to_csv('Example_IMC_2-results/Grid_points.csv')
print(imc.dictionary)


#output IMC
portion = 8
i = 0
j, Qiter_slice = imc.getQ_slice(i, portion)

tic1 = time.time()
"""
with open('Example_IMC_2-results/IMC_abstraction_matrix_{}.txt'.format(i+1), 'w') as f:
    for i, q in enumerate(Qiter_slice):
        f.write(str(i + j) + ': ')
        for k1, k2, k3 in imc.getrow(q):
            f.write(str(k1) + " " + str(k2) + " " + str(k3))
            f.write('; ')
        f.write('\n')
"""

with open('Example_IMC_2-results/IMC_abstraction_matrix_new_{}.txt'.format(i+1), 'w') as f:
    count = 0
    f.write(str(imc.N_matrix) + '\n')
    tic2 = time.time()
    for i, q in enumerate(Qiter_slice):#imc.getQ):
        row = list(imc.getrow(q))
        #f.write(str(i) + " ")
        for k in range(0, row[-1]):
            f.write(str(row[k][0]) + " " + str(row[k][1]) + " " + str(row[k][2]) + " ")
            #f.write('; ')
        f.write(str(imc.N_matrix) + '\n')
        count += row[-1]
    #f.write(str(imc.N_matrix - 1) + " 1 1 " + str(imc.N_matrix) + '\n')
    #count += 1

print('count=', count)

print('Computation time of 1/{0} portion of IMC abstraction = {1} sec'.format(portion, time.time() - tic1))
        




