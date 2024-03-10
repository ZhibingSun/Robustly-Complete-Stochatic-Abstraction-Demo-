#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:29:22 2023

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


X = [[1e-4, 1.2]]
precision = 8e-4
a = 0.2 # mean reversion rate
l = 0.05 # long-term mean
sigma = 0.3 # volatility
dt = 0.0001

L_f = 1 - a * dt
L_b = 50 * math.sqrt(dt) * sigma

# f_const = a * l * dt
# f = lambda x: L_f * x + f_const

# b_const = sigma * math.sqrt(dt)
# b = lambda x: lib.sqrt(x) * b_const


f = lambda x: x + a * (l - x) * dt
b = lambda x: sigma * lib.sqrt(x) * math.sqrt(dt)



imc = IMC(X, precision, f, b, L_f, L_b, use_fn_b=True, ws_dist_ratio=1.45)
print(imc.dictionary)

dirname = '/Users/z235sun/Desktop/Example_IMC_1-new'
if not os.path.isdir(dirname):
    os.mkdir(dirname)

#output IMC

portion =1
i = 0
# j, Qiter_slice = imc.getQ_slice(i, portion)

tic1 = time.time()

def main():

    with open(dirname + '/IMC_abstraction_matrix_new_{}.txt'.format(1), 'w') as f:
        count = 0
        
        # txt = ""
        # txt += f"{imc.N_matrix}\n"
        # # for i, q in enumerate(imc.idx_cube[:-1]):
        # for i, (q) in tqdm(enumerate(imc.idx_cube[:-1]),total = imc.N_matrix-1):
        # # for i, (q) in tqdm_gui(enumerate(imc.idx_cube[:-1]),leave=True):
        #     index, lower_bounds, upper_bounds = imc.output(q)
        #     for k in range(len(index)):
        #         txt += f"{index[k]} {lower_bounds[k]} {upper_bounds[k]} "
        #     txt += f"{imc.N_matrix}\n"
            
        #     # for k in range(len(index)):
        #     #     f.write(f"{index[k]} {lower_bounds[k]} {upper_bounds[k]} ")
        #     # # row = imc.output(q)
        #     # # f.write(row.tobytes())
        #     # f.write(f"{imc.N_matrix}\n")

        #     count += len(index)
        # txt += f"{imc.N_matrix-1} 1.0 1.0 {imc.N_matrix}\n"
        # f.write(txt)

        f.write(f"{imc.N_matrix}\n")
        # for i, q in enumerate(imc.idx_cube[:-1]):
        for i, (q) in tqdm(enumerate(imc.idx_cube[:-1]),total = imc.N_matrix-1):
        # for i, (q) in tqdm_gui(enumerate(imc.idx_cube[:-1]),leave=True):
            index, lower_bounds, upper_bounds = imc.output(q)
            
            # txt = ""
            # for k in range(len(index)):
            #     txt += f"{index[k]} {lower_bounds[k]} {upper_bounds[k]} "
            # txt += f"{imc.N_matrix}\n"
            # f.write(txt)
            
            for k in range(len(index)):
                f.write(f"{index[k]} {lower_bounds[k]} {upper_bounds[k]} ")
            
            # # row = imc.output(q)
            # # f.write(row.tobytes())
            f.write(f"{imc.N_matrix}\n")

            count += len(index)
        f.write(f"{imc.N_matrix-1} 1.0 1.0 {imc.N_matrix}\n")
        
        
        count += 1
    return count


# with open(dirname + '/IMC_abstraction_matrix_new_{}.txt'.format(i+1), 'w') as f:
#     count = 0
#     f.write(str(imc.N_matrix) + '\n')
#     # for i, q in enumerate(imc.idx_cube[:-1]):
#     for i, (q) in tqdm(enumerate(imc.idx_cube[:-1]),total = imc.N_matrix-1):
#     # for i, (q) in tqdm_gui(enumerate(imc.idx_cube[:-1]),leave=True):
#         row = imc.output(q)
#         for k in range(0, len(row)):
#             f.write(f"{row[k][0]} {row[k][1]} {row[k][2]} ")
#         f.write(f"{imc.N_matrix}\n")
#         count += len(row)
#     f.write(f"{imc.N_matrix-1} 1.0 1.0 {imc.N_matrix}\n")
#     count += 1

# with open(dirname + '/IMC_abstraction_matrix_new_{}.txt'.format(i+1), 'w') as f:
#     count = 0
#     rows = []
#     f.write(str(imc.N_matrix) + '\n')
#     # for i, q in enumerate(imc.idx_cube[:-1]):
#     for i, (q) in tqdm(enumerate(imc.idx_cube[:-1]),total = imc.N_matrix-1):
#     # for i, (q) in tqdm_gui(enumerate(imc.idx_cube[:-1]),leave=True):
#         rows.append(imc.output(q))
#         count += len(rows[i])
    
#     for i, (row) in tqdm(enumerate(rows),total = imc.N_matrix-1):
#     # for i, (q) in tqdm_gui(enumerate(imc.idx_cube),leave=True):
#         for k in range(0, len(row)):
#             f.write(f"{row[k][0]} {row[k][1]} {row[k][2]} ")
#         f.write(f"{imc.N_matrix}\n")
    
#     f.write(f"{imc.N_matrix-1} 1.0 1.0 {imc.N_matrix}\n")
#     count += 1

# print(imc.count, imc.err_max)
# print(imc.count)

count = main()
with open(dirname + '/label1_1.txt', 'w') as f:
    f.write(f"0 {imc.N_matrix} {count}\n")
    # f.write(str(count) + '\n')

# print(imc.count, imc.err_max)

print(f'Computation time of IMC abstraction = {time.time() - tic1} sec')

