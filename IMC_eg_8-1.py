
from src.IMC_Abstraction import IMC
import math
import time
import os 
from tqdm import tqdm
from tqdm.gui import tqdm_gui

dim = 1
dependency = 1
symmetry = 0
bound = 1
x = [0, bound]
f = lambda x: x
tag_f = 1
cif_f = 0

dt = 1.0 / (1 << 14)
b_const = math.sqrt(dt)
b = b_const
tag_b = 1
cif_b = 0
n = 1 << 16
# precision = 1e-4

# label_eps = 1e-3
# targets = [[-label_eps, label_eps]]

imc = IMC(dim, dependency, symmetry, x, f, tag_f, cif_f, b, tag_b, cif_b, n)

dirname = '/Users/z235sun/Desktop/IMC_eg_8-1'
if not os.path.isdir(dirname):
    os.mkdir(dirname)


tic1 = time.time()

def main():

    with open(dirname + '/imc82.txt'.format(1), 'w') as f:
        f.write(f"{imc.n_matrix}\n")
        # for i, q in enumerate(imc.index_cube[:-1]):
        for i, q in tqdm(enumerate(imc.idx_cube[:-2]),total = imc.n):
        # for i, q in tqdm_gui(enumerate(imc.idx_cube[:-1]),leave=True):
            
            index = imc.evaluate_itvl_prob(q)
            for k in range(len(index)):
                f.write(f"{index[k]} {imc.itvl_min_max[0][k]} {imc.itvl_min_max[1][k]} ")
            
            # index, lower_bounds, upper_bounds = imc.output(q)
            # for k in range(len(index)):
            #     f.write(f"{index[k]} {lower_bounds[k]} {upper_bounds[k]} ")
            
            f.write(f"{imc.n_matrix}\n")
            imc.m += len(index)
        # target, sink
        f.write(f"{imc.n} 1.0 1.0 {imc.n_matrix}\n{imc.n + 1} 1.0 1.0 {imc.n_matrix}\n")
        imc.m += 2

main()

print(imc)
# print(f'# of prob sinks > threshold = {imc.count}.')

with open(dirname + '/label82_1.txt', 'w') as f:
    
    f.write(f"{imc.n} {imc.n_matrix} {imc.m}\n")

print(f'Computation time of IMC abstraction = {time.time() - tic1} sec.\n')