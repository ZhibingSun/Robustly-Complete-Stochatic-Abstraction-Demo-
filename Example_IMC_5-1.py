#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:36:57 2023

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


X = [[-1, 1]]
precision = 1e-4  
dt = 1e-10

b_const = math.sqrt(dt)

f = lambda x: x 
b = lambda x: b_const

L_f = 1 
L_b = 0
cor = False

imc = IMC(X, precision, f, b, L_f, L_b, use_fn_b=True)
print(imc.dictionary)

dirname = '/Users/z235sun/Desktop/Example_IMC_5-new'
if not os.path.isdir(dirname):
    os.mkdir(dirname)

#output IMC

tic1 = time.time()

def main():

    with open(dirname + '/IMC_abstraction_matrix_new_{}.txt'.format(1), 'w') as f:
        count = 0
        f.write(f"{imc.N_matrix}\n")
        # for i, q in enumerate(imc.idx_cube[:-1]):
        for i, (q) in tqdm(enumerate(imc.idx_cube[:-1]),total = imc.N_matrix-1):
        # for i, (q) in tqdm_gui(enumerate(imc.idx_cube[:-1]),leave=True):
            index, lower_bounds, upper_bounds = imc.output(q)
            for k in range(len(index)):
                f.write(f"{index[k]} {lower_bounds[k]} {upper_bounds[k]} ")
            
            # # row = imc.output(q)
            # # f.write(row.tobytes())
            f.write(f"{imc.N_matrix}\n")

            count += len(index)
        f.write(f"{imc.N_matrix-1} 1.0 1.0 {imc.N_matrix}\n")
        count += 1
    return count

count = main()
with open(dirname + '/label5_1.txt', 'w') as f:
    tmp = math.floor((imc.N_matrix - 1) / 2)
    f.write(f"{tmp} {tmp + 1} {imc.N_matrix} {count}\n")
    # f.write(str(count) + '\n')

print(f'Computation time of IMC abstraction = {time.time() - tic1} sec')