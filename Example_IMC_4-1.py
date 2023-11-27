#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:05:04 2023

@author: ym
"""

from IMCclass_new import IMC
import Library as lib
import numpy as np
import time
import math
import pandas as pd
import os 



X = [[-1, 1], [-1, 1]]
precision = 0.18
dt =  0.81 #0.49 #0.81
eps = 0.01 #0.165 #0.01
L_f = 1
L_b = [[0, 0], [0, math.sqrt(dt) * eps]] 
cor = False


f = lambda x: [x[0], x[1] + (x[1]*x[0]-lib.power(x[1], 3)) * dt]
fp1 = lambda x: [1, 0]
fp2 = lambda x: [x[1], 1 + (x[0]-3*lib.sqr(x[1])) * dt]
fp = [fp1, fp2]
b = [[0, 0], [0, lambda x: x * math.sqrt(dt) * eps]] 

IMC.cherr(0.00001)
IMC.chws_dist(1.4)
imc = IMC(X, precision, f, b, L_f, L_b, cor, use_fn_f=True)



dirname = 'Example_IMC_4-new'
if not os.path.isdir('./' + dirname):
    os.mkdir(dirname)

#output basic information
pd.DataFrame(imc.dictionary).to_csv(dirname + '/Dictionary.csv')
pd.DataFrame({'Grid Point': list(imc.getQ)}).\
    to_csv(dirname + '/Grid_points.csv')
print(imc.dictionary)


#output IMC
portion = 1
i = 0
j, Qiter_slice = imc.getQ_slice(i, portion)

tic1 = time.time()
"""
with open('Example_IMC_4-results/IMC_abstraction_matrix_{}.txt'.format(i+1), 'w') as f:
    for i, q in enumerate(Qiter_slice):
        f.write(str(i + j) + ': ')
        for k1, k2, k3 in imc.getrow(q):
            f.write(str(k1) + " " + str(k2) + " " + str(k3))
            f.write('; ')
        f.write('\n')
"""

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
    f.write(str(imc.N_matrix - 1) + " 1 1 " + str(imc.N_matrix) + '\n')
    count += 1

print('count=', count)
print('Computation time of 1/{0} portion of IMC abstraction = {1} sec'.format(portion, time.time() - tic1))
"""


with open(dirname + '/label4_4.txt', 'w') as f:     
    for i in range(0, 76):
        x = 30 + 76 * i
        y = 14
        for j in range(x, x + y + 1):
            f.write(str(j) + " ")
    f.write(str(imc.N_matrix))