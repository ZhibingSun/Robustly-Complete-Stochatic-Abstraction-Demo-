from src.IMC_Abstraction import IMC
import math
import time


import os 
from tqdm import tqdm
from tqdm.gui import tqdm_gui

import numpy as np
import matplotlib.pyplot as plt

dim = 1
dependency = 1
symmetry = 1
bound = 1
x = [-bound, bound]
f = lambda x: x
tag_f = 1
cif_f = 0

dt = 1e-8
b_const = math.sqrt(dt)
b = b_const
tag_b = 1
cif_b = 0
precision = 1e-4


imc_list = []
imc_list.append(IMC(dim, dependency, symmetry, x, f, tag_f, cif_f, b, tag_b, cif_b, precision))
print(imc_list[-1])
precision /= 2
imc_list.append(IMC(dim, dependency, symmetry, x, f, tag_f, cif_f, b, tag_b, cif_b, precision))
print(imc_list[-1])

tic1 = time.time()

filenames = ["./quan_max_power_index61_1.txt", "./quan_min_power_index61_1.txt",
             "./quan_max_power_index62_1.txt", "./quan_min_power_index62_1.txt"]
colors = ["black", "blue", "black", "blue"]
labels = ["Max1", "Min1", "Max2", "Min2"]
linestyles = ["solid", "solid", "dashdot", "dashdot"]
imc_idx = [0, 0, 1, 1]
plt.figure(figsize=(50,10))
for i, f in enumerate(filenames):
    with open(f, "r") as fp:
        lines = [x[:-1].split("\t\t") for x in fp.readlines()]
        lines = list(filter(lambda x: len(x) > 1, lines))
        print(len(lines))
        lines = [[imc_list[imc_idx[i]].pt_partition[int(x)], float(y)] for x, y in lines]
        lines = np.array(lines)
        plt.plot(lines[:,0], lines[:,1], label=labels[i], color=colors[i], linestyle=linestyles[i])
plt.legend()

# plt.show()
plt.savefig("/Users/z235sun/Desktop/IMC_eg_5-1/IMC_eg_6-1.pdf", dpi=5000)

plt.yscale("log")
# # plt.show()
plt.savefig("/Users/z235sun/Desktop/IMC_eg_5-1/IMC_eg_6-1_log.pdf", dpi=5000)

print(f'Computation time of IMC plot = {time.time() - tic1} sec')
