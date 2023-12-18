#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 01 10:19:52 2023

@author: ym
"""

import src.Library as lib
from src.IMCclass_new import IMC
import numpy as np
import time
import math
import pandas as pd
import os 
from tqdm import tqdm
from tqdm.gui import tqdm_gui


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

imc = IMC(X, precision, f, b, L_f, L_b, cor=cor, fp=fp)
#imc.chws_dist(0.001)

"""
dirname = 'Example_IMC_2-results'
if not os.path.isdir('./' + dirname):
    os.mkdir(dirname)
"""
#output basic information

print(imc.dictionary)


#output IMC
portion = 8
i = 0
# j, Qiter_slice = imc.getQ_slice(i, portion)

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
dirname = '/Users/z235sun/Desktop/Example_IMC_2-new'
if not os.path.isdir(dirname):
    os.mkdir(dirname)
with open(dirname + '/IMC_abstraction_matrix_new_{}.txt'.format(i+1), 'w') as f:
    count = 0
    f.write(str(imc.N_matrix) + '\n')
    # for i, q in enumerate(imc.idx_cube[:-1]):
    for i, (q) in tqdm(enumerate(imc.idx_cube[:-1]),total = imc.N_matrix-1):
    # for i, (q) in tqdm_gui(enumerate(imc.idx_cube[:-1]),leave=True):
        index, lower_bounds, upper_bounds = imc.output(q)
        for k in range(len(index)):
            f.write(f"{index[k]} {lower_bounds[k]} {upper_bounds[k]} ")
        # row = imc.output(q)
        # f.write(row.tobytes())
        f.write(f"{imc.N_matrix}\n")
        count += len(index)
    f.write(f"{imc.N_matrix-1} 1.0 1.0 {imc.N_matrix}\n")
    count += 1
# print(imc.count)
print('count=', count)

print(f'Computation time of IMC abstraction = {time.time() - tic1} sec')
        




