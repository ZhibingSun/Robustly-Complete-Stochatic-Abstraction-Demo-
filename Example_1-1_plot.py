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
import matplotlib.pyplot as plt

X = [[0.0001, 1.2]]
precision = 8e-4
a = 0.2 # mean reversion rate
l = 0.05 # long-term mean
sigma = 0.3 # volatility
dt = 0.0001


f = lambda x: x + a * (l - x) * dt
b = lambda x: sigma * lib.sqrt(x) * math.sqrt(dt)

L_f = 1 - a * dt
L_b = 50 * math.sqrt(dt) * sigma

imc_list = []
imc_list.append(IMC(X, precision, f, b, L_f, L_b, use_fn_b=True, ws_dist_ratio=1.45))
print(imc_list[-1].dictionary)
precision = 8e-5
imc_list.append(IMC(X, precision, f, b, L_f, L_b, use_fn_b=True, ws_dist_ratio=1.45))
print(imc_list[-1].dictionary)

tic1 = time.time()

filenames = ["./quan_max_power_index11_1.txt", "./quan_min_power_index11_1.txt",
             "./quan_max_power_index12_1.txt", "./quan_min_power_index12_1.txt"]
colors = ["black", "blue", "black", "blue"]
labels = ["Max1", "Min1", "Max2", "Min2"]
linestyles = ["solid", "solid", "dashdot", "dashdot"]
imc_idx = [0, 0, 1, 1]
plt.figure(figsize=(50,10))
for i, f in enumerate(filenames):
    with open(f, "r") as fp:
        lines = [x[:-1].split("\t") for x in fp.readlines()]
        lines = list(filter(lambda x: len(x) > 1, lines))
        print(len(lines))
        lines = [[imc_list[imc_idx[i]].pt_partition[int(x)], float(y)] for x, z, y in lines]
        lines = np.array(lines)
        plt.plot(lines[:,0], lines[:,1], label=labels[i], color=colors[i], linestyle=linestyles[i])
plt.legend()




# plt.show()
plt.savefig("/Users/z235sun/Desktop/Example_IMC_1-new/Example_1-1.pdf", dpi=5000)

plt.yscale("log")
# # plt.show()
plt.savefig("/Users/z235sun/Desktop/Example_IMC_1-new/Example_1-1_log.pdf", dpi=5000)

print(f'Computation time of IMC plot = {time.time() - tic1} sec')



# dirname = '/Users/z235sun/Desktop/Example_IMC_1-new'
# if not os.path.isdir(dirname):
#     os.mkdir(dirname)

# #output IMC










