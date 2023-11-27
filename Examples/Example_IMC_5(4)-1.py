#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:36:57 2023

@author: ym
"""

from IMCclass_new import IMC
import Library as lib
import numpy as np
import time
import math
import pandas as pd
import os 



X = [[-1, 1]]
precision = 0.0001  
dt = 0.0000000001


f = lambda x: x 
b = lambda x: math.sqrt(dt)

L_f = 1 
L_b = 0
cor = False

IMC.chws_dist(1.3)
imc = IMC(X, precision, f, b, L_f, L_b, use_fn_b=True)
print(imc.dictionary)


dirname = 'Example_IMC_5（4）'


if not os.path.isdir('./' + dirname):
    os.mkdir(dirname)


pd.DataFrame(imc.dictionary).to_csv(dirname + '/Dictionary.csv')
pd.DataFrame({'Grid Point': list(imc.getQ)}).\
    to_csv(dirname + '/Grid_points.csv')


#output IMC

portion =10000
i = 0
j, Qiter_slice = imc.getQ_slice(i, portion)

tic1 = time.time()

"""
with open(dirname + '/IMC_abstraction_matrix_new_{}.txt'.format(i+1), 'w') as f:
    count = 0
    f.write(str(imc.N_matrix) + '\n')
    for i, q in enumerate(Qiter_slice):
        f.write(str(i + j) + " ")
        for k1, k2, k3 in imc.getrow(q):
            f.write(str(k1) + " " + str(k2) + " " + str(k3)) + " "
            #f.write('; ')
        f.write('\n')
"""

with open(dirname + '/IMC_abstraction_matrix_new_{}.txt'.format(i+1), 'w') as f:
    count = 0
    f.write(str(imc.N_matrix) + '\n')
    for i, q in enumerate(Qiter_slice):
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

print('Computation time of IMC abstraction = {0} sec'.format(time.time() - tic1))