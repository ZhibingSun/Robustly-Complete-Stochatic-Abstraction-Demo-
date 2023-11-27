#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:29:22 2023

@author: ym
"""

from IMCclass_new import IMC
import Library as lib
import numpy as np
import time
import math
import pandas as pd
import os 
from tqdm import tqdm


X = [[0.0001, 1.2]]
precision = 0.0008
a = 0.2 # mean reversion rate
l = 0.05 # long-term mean
sigma = 0.3 # volatility
dt = 0.0001


f = lambda x: x + a * (l - x) * dt
b = lambda x: sigma * lib.sqrt(x) * math.sqrt(dt)

L_f = 1 - a * dt
L_b = 50 * math.sqrt(dt) * sigma

IMC.chws_dist(1.45)
imc = IMC(X, precision, f, b, L_f, L_b, use_fn_b=True)
print(imc.dictionary)


dirname = 'Example_IMC_1-new'
if not os.path.isdir('./' + dirname):
    os.mkdir(dirname)

pd.DataFrame(imc.dictionary).to_csv(dirname + '/Dictionary.csv')
pd.DataFrame({'Grid Point': list(imc.getQ)}).\
    to_csv(dirname + '/Grid_points.csv')


#output IMC

portion =1
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
    for i, q in enumerate(imc.getQ):
    # for i, (q) in tqdm(enumerate(imc.getQ):
        row = list(imc.getrow(q))
        #f.write(str(i) + " ")
        for k in range(0, row[-1]):
            f.write(str(row[k][0]) + " " + str(row[k][1]) + " " + str(row[k][2]) + " ")
            #f.write('; ')
        f.write(str(imc.N_matrix) + '\n')
        count += row[-1]
    f.write(str(imc.N_matrix - 1) + " 1 1 " + str(imc.N_matrix) + '\n')
    count += 1

with open(dirname + '/label1_1.txt', 'w') as f:
    f.write(str(count) + '\n')

print('Computation time of IMC abstraction = {0} sec'.format(time.time() - tic1))

