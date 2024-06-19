import math
import numpy as np
from scipy.special import ndtr
import itertools

class Interval:
    def __init__(self, itvl):
        self.itvl = itvl

    def __repr__(self) -> str:
        return f'[{self.itvl[0]:.7e}, {self.itvl[1]:.7e}]'
    
    ### Interval function operation overload starts.
    
    # +self.itvl
    def __pos__(self):
        return self
    
    # self.itvl + x
    def __add__(self, x):
        if isinstance(x, Interval):
            return Interval([self.itvl[0] + x.itvl[0], self.itvl[1] + x.itvl[1]])
        # x is scalar
        return Interval([self.itvl[0] + x, self.itvl[1] + x])

    # x + self.itvl
    def __radd__(self, x): # x is scalar
        return Interval([self.itvl[0] + x, self.itvl[1] + x])

    # -self.itvl
    def __neg__(self):
        return Interval([-self.itvl[1], -self.itvl[0]])

    # self.itvl - x
    def __sub__(self, x):
        if isinstance(x, Interval):
            return Interval([self.itvl[0] - x.itvl[1], self.itvl[1] - x.itvl[0]])
        # x is scalar
        return Interval([self.itvl[0] - x, self.itvl[1] - x])

    # x - self.itvl
    def __rsub__(self, x): # x is scalar
        return Interval([x - self.itvl[1], x - self.itvl[0]])

    # self.itvl * x
    def __mul__(self, x):
        if isinstance(x, Interval):
            tmp = [self.itvl[0] * x.itvl[0], self.itvl[0] * x.itvl[1],
                   self.itvl[1] * x.itvl[0], self.itvl[1] * x.itvl[1]]
            return Interval([min(*tmp), max(*tmp)])
        # x is scalar
        if x >= 0:
            return Interval([x * self.itvl[0], x * self.itvl[1]])
        # x < 0
        return Interval([x * self.itvl[1], x * self.itvl[0]])

    # x * self.itvl
    def __rmul__(self, x): # x is scalar
        if x >= 0:
            return Interval([x * self.itvl[0], x * self.itvl[1]])
        # x < 0
        return Interval([x * self.itvl[1], x * self.itvl[0]])

    # self.itvl / x
    def __truediv__(self, x):
        if isinstance(x, Interval):
            if x.itvl[0] > 0 or x.itvl[1] < 0:
                reciprocal = [1 / x.itvl[1], 1 / x.itvl[0]]
            elif x.itvl[0] == 0 and x.itvl[1] == 0:
                reciprocal = [math.nan, math.nan]
            elif x.itvl[0] == 0 and x.itvl[1] > 0:
                reciprocal = [1 / x.itvl[1], math.inf]
            elif x.itvl[0] < 0 and x.itvl[1] == 0:
                reciprocal = [-math.inf, 1 / x.itvl[0]]
            else: # x.itvl[0] < 0 and x.itvl[1] > 0
                reciprocal = [-math.inf, math.inf]
            tmp = [self.itvl[0] * reciprocal[0], self.itvl[0] * reciprocal[1],
                   self.itvl[1] * reciprocal[0], self.itvl[1] * reciprocal[1]]
            return Interval([min(*tmp), max(*tmp)])
        # x is scalar
        reciprocal = 1 / x
        if reciprocal >= 0:
            return Interval([reciprocal * self.itvl[0], reciprocal * self.itvl[1]])
        # x < 0
        return Interval([reciprocal * self.itvl[1], reciprocal * self.itvl[0]])

    # x / self.itvl
    def __rtruediv__(self, x): # x is scalar
        if self.itvl[0] > 0 or self.itvl[1] < 0:
            reciprocal = [1 / self.itvl[1], 1 / self.itvl[0]]
        elif self.itvl[0] == 0 and self.itvl[1] == 0:
            reciprocal = [math.nan, math.nan]
        elif self.itvl[0] == 0 and self.itvl[1] > 0:
            reciprocal = [1 / self.itvl[1], math.inf]
        elif self.itvl[0] < 0 and self.itvl[1] == 0:
            reciprocal = [-math.inf, 1 / self.itvl[0]]
        else: # x.itvl[0] < 0 and x.itvl[1] > 0
            reciprocal = [-math.inf, math.inf]
        if x >= 0:
            return Interval([x * reciprocal[0], x * reciprocal[1]])
        # x < 0
        return Interval([x * reciprocal[1], x * reciprocal[0]])

    ### Interval function operation overload ends.

    ### Elemantary interval function overload starts.
    
    def union(x, y):
        return Interval([min(x.itvl[0], y.itvl[0]), max(x.itvl[1], y.itvl[1])])

    def intersection(x, y):
        tmp = [max(x.itvl[0], y.itvl[0]), min(x.itvl[1], y.itvl[1])]
        if tmp[0] <= tmp[1]:
            return Interval(tmp)
        return Interval([math.nan, math.nan])

    def sqr(x):
        if isinstance(x, Interval):
            if x.itvl[0] > 0:
                return Interval([x.itvl[0]**2, x.itvl[1]**2])
            elif x.itvl[1] < 0:
                return Interval([x.itvl[1]**2, x.itvl[0]**2])
            else: # x.itvl[0] <= 0 and x.itvl[1] >= 0
                return Interval([0, max(-x.itvl[0], x.itvl[1])**2])
        # x is scalar
        return x**2
    
    def sqrt(x):
        if isinstance(x, Interval):
            if x.itvl[0] > 0:
                return Interval([math.sqrt(x.itvl[0]), math.sqrt(x.itvl[1])])
            elif x.itvl[1] >= 0:
                return Interval([0, math.sqrt(x.itvl[1])])
            else: # x.itvl[1] < 0
                return Interval([math.nan, math.nan])
        # x is scalar
        return math.sqrt(x)

    def exp(x):
        if isinstance(x, Interval):
            return Interval([math.exp(x.itvl[0]), math.exp(x.itvl[1])])
        # x is scalar
        return math.exp(x)

    def log(x):
        if isinstance(x, Interval):
            if x.itvl[0] > 0:
                return Interval([math.log(x.itvl[0]), math.log(x.itvl[1])])
            elif x.itvl[1] > 0:
                return Interval([-math.inf, math.log(x.itvl[1])])
            else: # x.itvl[1] <= 0
                return Interval([math.nan, math.nan])
        # x is scalar
        return math.log(x)

    def sin(x):
        if isinstance(x, Interval):
            sin_itvl = [-1,1]
            const = 0.5 / math.pi # 1 / (2 * pi)
            # determine upper bound sin_itvl[1]
            tmp = [x.itvl[0] * const - 0.25, x.itvl[1] * const - 0.25] # [x] / (2 * pi) - 1/4
            if math.ceil(tmp[0]) > math.floor(tmp[1]):
                tmp1 = [math.sin(x.itvl[0]), math.sin(x.itvl[1])]
                sin_itvl[1] = max(*tmp1)
            # determine lower bound sin_itvl[0]
            # [x] / (2 * pi) + 1/4
            tmp[0] += 0.5
            tmp[1] += 0.5
            if math.ceil(tmp[0]) > math.floor(tmp[1]):
                if sin_itvl[1] == 1:
                    tmp1 = [math.sin(x.itvl[0]), math.sin(x.itvl[1])]    
                # else tmp1 has been computed above
                sin_itvl[0] = min(*tmp1)
            return Interval(sin_itvl)
        # x is scalar
        return math.sin(x)

    def cos(x):
        if isinstance(x, Interval):
            cos_itvl = [-1,1]
            const = 0.5 / math.pi # 1 / (2 * pi)
            # determine upper bound cos_itvl[1]
            tmp = [x.itvl[0] * const, x.itvl[1] * const] # [x] / (2 * pi)
            if math.ceil(tmp[0]) > math.floor(tmp[1]):
                tmp1 = [math.cos(x.itvl[0]), math.cos(x.itvl[1])]
                cos_itvl[1] = max(*tmp1)
            # determine lower bound cos_itvl[0]
            # [x] / (2 * pi) - 1/2
            tmp[0] -= 0.5
            tmp[1] -= 0.5
            if math.ceil(tmp[0]) > math.floor(tmp[1]):
                if cos_itvl[1] == 1:
                    tmp1 = [math.cos(x.itvl[0]), math.cos(x.itvl[1])]    
                # else tmp1 has been computed above
                cos_itvl[0] = min(*tmp1)
            return Interval(cos_itvl)
        # x is scalar
        return math.cos(x)

    # May not be continuous, so not implemented.
    # def tan(x):
    #     pass

    # def __(self, x):
        
    #     pass

    ### Elemantary interval function overload end

    # natural inclusion function
    # def __nif(self, dim, f, x)

class IMC:
    def __init__(self, dim, dependency, symmetry, x, f, tag_f, cif_f,\
                 b, tag_b, cif_b, n, precision=1e-4,\
                 kappa_coefficient=0.50,eps_margin=0.999,\
                 ball_coefficient=1,ws_dist_ratio = 1.3, err=1e-8):

        self.dim = dim
        self.dependency = dependency
        self.symmetry = symmetry
        self.x = x
        self.f = f
        self.tag_f = tag_f
        self.cif_f = cif_f
        self.b = b
        self.tag_b = tag_b
        self.cif_b = cif_b

        # parameters for completeness
        # self.precision = precision
        # self.kappa_coefficient = kappa_coefficient
        # self.eps_margin = eps_margin
        # self.ball_coefficient = ball_coefficient
        # self.ws_dist_ratio = ws_dist_ratio
        # self.err = err

        #This is used for a direct calculation of eta
        # self.eps = self.precision * (self.eps_margin - self.kappa_coefficient)

        #Determine the 'inflation' precision for the inclusion functions
        # self.kappa = self.kappa_coefficient * self.precision
        
        # eta_nominal = self.eps / (self.ws_dist_ratio + 2 * self.ball_coefficient)


        if self.dim == 1:
            # don't need dependency 
            self.wid = self.x[1] - self.x[0]
            self.mid = (self.x[1] + self.x[0]) * 0.5
            self.vol = self.wid
            # self.n = math.ceil(self.wid / eta_nominal)
            self.n = n
            
            # self.n = 10278
            
            if self.symmetry == 1 and not (self.n & 1):
                self.n += 1
            elif self.symmetry == 2 and (self.n & 1):
                self.n += 1

            self.eta = self.wid / self.n
            self.n_matrix = self.n + 2

            # tmp = self.ws_dist_ratio * self.eta
            # self.beta = tmp 
            # if self.L_b == 0 \
            # else tmp / math.sqrt(2)

            self.idx_cube = list(range(self.n_matrix))
            if self.symmetry == 0:
                self.pt_partition = np.linspace(self.x[0], self.x[1], self.n + 1)
            elif self.symmetry == 1:
                self.pt_partition = np.zeros(self.n + 1)
                tmp = (self.n >> 1) + 1
                self.pt_partition[tmp:] = np.array(range(tmp)) * self.eta + (self.eta / 2)
                self.pt_partition[:tmp] = np.flip(-self.pt_partition[tmp:])
                self.pt_partition += self.mid
            else: # self.symmetry == 2
                tmp = self.n >> 1
                self.pt_partition = np.array(range(-tmp, tmp + 1)) * self.eta + self.mid
            
            # if tag_b == 1: # b is const
            self.cdf = np.zeros(self.n_matrix)
            # else: # tag_b > 1 b(x) is not const
            
            self.itvl_tmp = np.zeros((2, self.n + 2))
            self.itvl_min_max = np.zeros((2, self.n + 2))
            self.rel_l = 0
            self.rel_r = 1
            self.abs_l = 0
            self.abs_r = 1
            self.cdf_tmp = np.zeros(2)
            self.idx_tmp = np.zeros(2)
            
            self.target_tmp = np.zeros(2)
            self.target_min_max = np.zeros(2)
            
            self.sink_tmp = np.zeros(2)
            self.sink_min_max = np.zeros(2)
            self.tag_target_sink = 0
            self.sum_l = 0
            self.sum_u = 0
            self.ndtr_rel_peak = 3.4e-14
            self.ndtr_rel_u = 1 / (1 - self.ndtr_rel_peak)
            self.ndtr_rel_l = 1 / (1 + self.ndtr_rel_peak)

        else: # self.dim > 1

            pass
        
        
        # self.lower_threshold = 1 - self.err - self.beta
        self.count = 0
        self.m = 0
        self.err_max = 0
        self.threshold = np.finfo(float).eps * 0.5

    def __repr__(self) -> str:
        return f'dim = {self.dim}, dependency = {self.dependency}, '\
             + f'symmetry = {self.symmetry}, x = {self.x}, n = {self.n}, '\
             + f'n_matrix = {self.n_matrix}, m = {self.m}, '\
             + f'(m - 2) / n = {(self.m - 2) // self.n}, wid = {self.wid}, '\
             + f'mid = {self.mid}, eta = {self.eta:.17e}, '\
             + f'threshold = {self.threshold}.'
            # + f'precision = {self.precision:.7e}, beta = {self.beta}, '\
             



    def evaluate_itvl_prob(self, idx):
        if self.dim == 1:
            x_i = [self.pt_partition[idx],self.pt_partition[idx + 1]]
            wid_i = x_i[1] - x_i[0]
            
            if self.tag_f == 1 and self.tag_b == 1:
                # version 0
                # can have root of all evil here
                # ndtr((self.pt_partition - x_i[0]) / self.b, out=self.cdf)
                # self.itvl_tmp[0][:-1] = self.cdf[1:] - self.cdf[:-1]
                # self.itvl_tmp[0][-1] = 1 - self.cdf[-1] + self.cdf[0]

                # ndtr((self.pt_partition - x_i[1]) / self.b, out=self.cdf)
                # self.itvl_tmp[1][:-1] = self.cdf[1:] - self.cdf[:-1]
                # self.itvl_tmp[1][-1] = 1 - self.cdf[-1] + self.cdf[0]

                # self.itvl_min_max[0] = np.min(self.itvl_tmp, axis=0)
                # self.itvl_min_max[1] = np.max(self.itvl_tmp, axis=0)

                # self.itvl_min_max[1][idx] = 2 * ndtr((x_i[1] - x_i[0]) * 0.5 / self.b) - 1

                # if self.mid > x_i[0] and self.mid < x_i[1]:
                #     self.itvl_min_max[0][-1] = 2 - 2 * ndtr(self.wid * 0.5 / self.b)


                # version 1
                # ndtr((self.pt_partition - x_i[0]) / self.b, out=self.cdf)
                # self.itvl_tmp[0][:-1] = self.cdf[1:] - self.cdf[:-1]
                # self.itvl_tmp[0][-1] = 1 - self.cdf[-1] + self.cdf[0]
                

                # self.itvl_min_max[1][:idx] = self.itvl_tmp[0][:idx]
                # self.itvl_min_max[0][idx+1:-1] = self.itvl_tmp[0][idx+1:-1]

                # ndtr((self.pt_partition - x_i[1]) / self.b, out=self.cdf)
                # self.itvl_tmp[1][:-1] = self.cdf[1:] - self.cdf[:-1]
                # self.itvl_tmp[1][-1] = 1 - self.cdf[-1] + self.cdf[0]

                # self.itvl_min_max[0][:idx] = self.itvl_tmp[1][:idx]
                # self.itvl_min_max[1][idx+1:-1] = self.itvl_tmp[1][idx+1:-1]

                # self.itvl_min_max[0][idx] = min(self.itvl_tmp[0][idx], self.itvl_tmp[1][idx])
                # self.itvl_min_max[1][idx] = 2 * ndtr((x_i[1] - x_i[0]) * 0.5 / self.b) - 1
                

                # if self.mid >= x_i[1]:
                #     self.itvl_min_max[0][-1] = self.itvl_tmp[1][-1]
                #     self.itvl_min_max[1][-1] = self.itvl_tmp[0][-1]
                # elif self.mid <= x_i[0]:
                #     self.itvl_min_max[0][-1] = self.itvl_tmp[0][-1]
                #     self.itvl_min_max[1][-1] = self.itvl_tmp[1][-1]
                # else: # self.mid > x_i[0] and self.mid < x_i[1]:
                #     self.itvl_min_max[0][-1] = 2 - 2 * ndtr(self.wid * 0.5 / self.b)
                #     self.itvl_min_max[1][-1] = max(self.itvl_tmp[0][-1], self.itvl_tmp[1][-1])

                # version 2
                # ndtr((self.pt_partition - x_i[0]) / self.b, out=self.cdf)

                # self.itvl_tmp[0][idx] = self.cdf[idx+1] - self.cdf[idx]
                # self.itvl_tmp[0][-1] = 1 - self.cdf[-1] + self.cdf[0]
                
                
                # self.itvl_min_max[1][:idx] = self.cdf[1:idx+1] - self.cdf[:idx]
                # self.itvl_min_max[0][idx+1:-1] = self.cdf[idx+2:] - self.cdf[idx+1:-1]

                # ndtr((self.pt_partition - x_i[1]) / self.b, out=self.cdf)
                
                # self.itvl_tmp[1][idx] = self.cdf[idx+1] - self.cdf[idx]
                # self.itvl_tmp[1][-1] = 1 - self.cdf[-1] + self.cdf[0]

                # self.itvl_min_max[0][:idx] = self.cdf[1:idx+1] - self.cdf[:idx]
                # self.itvl_min_max[1][idx+1:-1] = self.cdf[idx+2:] - self.cdf[idx+1:-1]


                # self.itvl_min_max[0][idx] = min(self.itvl_tmp[0][idx], self.itvl_tmp[1][idx])
                # self.itvl_min_max[1][idx] = 2 * ndtr((x_i[1] - x_i[0]) * 0.5 / self.b) - 1
                

                # if self.mid >= x_i[1]:
                #     self.itvl_min_max[0][-1] = self.itvl_tmp[1][-1]
                #     self.itvl_min_max[1][-1] = self.itvl_tmp[0][-1]
                # elif self.mid <= x_i[0]:
                #     self.itvl_min_max[0][-1] = self.itvl_tmp[0][-1]
                #     self.itvl_min_max[1][-1] = self.itvl_tmp[1][-1]
                # else: # self.mid > x_i[0] and self.mid < x_i[1]:
                #     self.itvl_min_max[0][-1] = 2 - 2 * ndtr(self.wid * 0.5 / self.b)
                #     self.itvl_min_max[1][-1] = max(self.itvl_tmp[0][-1], self.itvl_tmp[1][-1])



                # This is just to test bound of sink
                # self.cdf[0] = ndtr((self.pt_partition[0] - x_i[0]) / self.b)
                # self.cdf[-1] = ndtr((self.pt_partition[-1] - x_i[0]) / self.b)

                # # self.itvl_tmp[0][idx] = self.cdf[idx+1] - self.cdf[idx]
                # self.itvl_tmp[0][-1] = 1 - self.cdf[-1] + self.cdf[0]
                
                
                # # self.itvl_min_max[1][:idx] = self.cdf[1:idx+1] - self.cdf[:idx]
                # # self.itvl_min_max[0][idx+1:-1] = self.cdf[idx+2:] - self.cdf[idx+1:-1]

                # self.cdf[0] = ndtr((self.pt_partition[0] - x_i[1]) / self.b)
                # self.cdf[-1] = ndtr((self.pt_partition[-1] - x_i[1]) / self.b)
                
                # # self.itvl_tmp[1][idx] = self.cdf[idx+1] - self.cdf[idx]
                # self.itvl_tmp[1][-1] = 1 - self.cdf[-1] + self.cdf[0]

                # # self.itvl_min_max[0][:idx] = self.cdf[1:idx+1] - self.cdf[:idx]
                # # self.itvl_min_max[1][idx+1:-1] = self.cdf[idx+2:] - self.cdf[idx+1:-1]


                # # self.itvl_min_max[0][idx] = min(self.itvl_tmp[0][idx], self.itvl_tmp[1][idx])
                # # self.itvl_min_max[1][idx] = 2 * ndtr((x_i[1] - x_i[0]) * 0.5 / self.b) - 1
                

                # if self.mid >= x_i[1]:
                #     self.itvl_min_max[0][-1] = self.itvl_tmp[1][-1]
                #     self.itvl_min_max[1][-1] = self.itvl_tmp[0][-1]
                # elif self.mid <= x_i[0]:
                #     self.itvl_min_max[0][-1] = self.itvl_tmp[0][-1]
                #     self.itvl_min_max[1][-1] = self.itvl_tmp[1][-1]
                # else: # self.mid > x_i[0] and self.mid < x_i[1]:
                #     self.itvl_min_max[0][-1] = 2 - 2 * ndtr(self.wid * 0.5 / self.b)
                #     self.itvl_min_max[1][-1] = max(self.itvl_tmp[0][-1], self.itvl_tmp[1][-1])
                # if self.itvl_min_max[1][-1] != 0:
                #     self.count += 1
                #     print(f'{idx}: {self.itvl_min_max[0][-1]}, {self.itvl_min_max[1][-1]}')



                # binary search version starts version 1
                # determine abs_l and rel_l
                # self.abs_l = idx - self.rel_l
                # ndtr((self.pt_partition[[self.abs_l, self.abs_l + 1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                # if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold: # move left
                #     while self.abs_l - 1 >= 0:
                #         ndtr((self.pt_partition[[self.abs_l - 1, self.abs_l]] - x_i[0]) / self.b, out = self.cdf_tmp)
                #         if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #             self.abs_l -= 1
                #             # self.count += 1
                #         else:
                #             break
                # else: # self.cdf_tmp[1] - self.cdf_tmp[0] <= self.threshold move right
                #     while 1 :
                #         self.abs_l += 1
                #         # self.count += 1
                #         ndtr((self.pt_partition[[self.abs_l, self.abs_l + 1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                #         if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #             break
                # self.rel_l = idx - self.abs_l
                # # determine abs_r and rel_r
                # self.abs_r = idx + self.rel_r
                # if self.abs_r > self.n:
                #     self.abs_r -= 1
                # else:
                #     ndtr((self.pt_partition[[self.abs_r - 1, self.abs_r]] - x_i[1]) / self.b, out = self.cdf_tmp)
                #     if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold: # move right
                #         while self.abs_r + 1 <= self.n:
                #             ndtr((self.pt_partition[[self.abs_r, self.abs_r + 1]] - x_i[1]) / self.b, out = self.cdf_tmp)
                #             if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #                 self.abs_r += 1
                #                 # self.count += 1
                #             else:
                #                 break
                #     else: # self.cdf_tmp[1] - self.cdf_tmp[0] <= self.threshold move left
                #         while 1 :
                #             self.abs_r -= 1
                #             # self.count += 1
                #             ndtr((self.pt_partition[[self.abs_r - 1, self.abs_r]] - x_i[1]) / self.b, out = self.cdf_tmp)
                #             if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #                 break
                # self.rel_r = self.abs_r - idx  

                
                # ndtr((self.pt_partition[[0, -1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                # self.sink_tmp[0] = 1 - self.cdf_tmp[1] + self.cdf_tmp[0]

                # ndtr((self.pt_partition[[0, -1]] - x_i[1]) / self.b, out = self.cdf_tmp)
                # self.sink_tmp[1] = 1 - self.cdf_tmp[1] + self.cdf_tmp[0]

                # # if self.mid >= x_i[1]:
                # #     self.sink_min_max[0] = self.sink_tmp[1]
                # #     self.sink_min_max[1] = self.sink_tmp[0]
                # # elif self.mid <= x_i[0]:
                # #     self.sink_min_max[0] = self.sink_tmp[0]
                # #     self.sink_min_max[1] = self.sink_tmp[1]
                # # else: # self.mid > x_i[0] and self.mid < x_i[1]:
                # #     self.itvl_min_max[0] = 2 - 2 * ndtr(self.wid * 0.5 / self.b)
                # #     self.sink_min_max[1] = max(self.sink_tmp)

                
                # tmp = self.abs_r - self.abs_l
                # # cdf = np.zeros(tmp + 1)
                # # itvl_tmp = np.zeros((2, tmp))
                # ndtr((self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[0]) / self.b, out=self.cdf[:tmp+1])
                # self.itvl_tmp[0][:tmp] = self.cdf[1:tmp+1] - self.cdf[:tmp]
                # ndtr((self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[1]) / self.b, out=self.cdf[:tmp+1])
                # self.itvl_tmp[1][:tmp] = self.cdf[1:tmp+1] - self.cdf[:tmp]

                

                # self.sink_min_max[1] = max(self.sink_tmp)
                # if self.sink_min_max[1] > self.threshold:
                #     # itvl_min_max = np.zeros((2, tmp + 1))
                #     if self.mid <= x_i[0] or self.mid >= x_i[1]:
                #         self.itvl_min_max[0][tmp] = min(self.sink_tmp)    
                #     else:
                #         self.itvl_min_max[0][tmp] = 2 - 2 * ndtr(self.wid * 0.5 / self.b)
                #     self.itvl_min_max[1][tmp] = self.sink_min_max[1]
                #     index = list(range(self.abs_l,self.abs_r + 1))
                #     index[-1] = self.n
                # else: # self.sink_min_max[1] <= self.threshold
                #     # itvl_min_max = np.zeros((2, tmp))
                #     index = list(range(self.abs_l,self.abs_r))
                # self.itvl_min_max[0][:tmp] = np.min(self.itvl_tmp[:,:tmp], axis=0)
                # self.itvl_min_max[1][:tmp] = np.max(self.itvl_tmp[:,:tmp], axis=0)
                # self.itvl_min_max[1][self.rel_l] = 2 * ndtr((x_i[1] - x_i[0]) * 0.5 / self.b) - 1
                
                # return index
                # binary search version ends version 1

                # binary search version starts version 2
                # determine abs_l and rel_l
                # self.abs_l = idx - self.rel_l
                # ndtr((self.pt_partition[[self.abs_l, self.abs_l + 1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                # if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold: # move left
                #     while self.abs_l - 1 >= 0:
                #         ndtr((self.pt_partition[[self.abs_l - 1, self.abs_l]] - x_i[0]) / self.b, out = self.cdf_tmp)
                #         if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #             self.abs_l -= 1
                #             # self.count += 1
                #         else:
                #             break
                # else: # self.cdf_tmp[1] - self.cdf_tmp[0] <= self.threshold move right
                #     while 1 :
                #         self.abs_l += 1
                #         # self.count += 1
                #         ndtr((self.pt_partition[[self.abs_l, self.abs_l + 1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                #         if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #             break
                # self.rel_l = idx - self.abs_l
                # # determine abs_r and rel_r
                # self.abs_r = idx + self.rel_r
                # if self.abs_r >self.n:
                #     self.abs_r -= 1
                # else:
                #     ndtr((self.pt_partition[[self.abs_r - 1, self.abs_r]] - x_i[1]) / self.b, out = self.cdf_tmp)
                #     if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold: # move right
                #         while self.abs_r + 1 <= self.n:
                #             ndtr((self.pt_partition[[self.abs_r, self.abs_r + 1]] - x_i[1]) / self.b, out = self.cdf_tmp)
                #             if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #                 self.abs_r += 1
                #                 # self.count += 1
                #             else:
                #                 break
                #     else: # self.cdf_tmp[1] - self.cdf_tmp[0] <= self.threshold move left
                #         while 1 :
                #             self.abs_r -= 1
                #             # self.count += 1
                #             ndtr((self.pt_partition[[self.abs_r - 1, self.abs_r]] - x_i[1]) / self.b, out = self.cdf_tmp)
                #             if self.cdf_tmp[1] - self.cdf_tmp[0] > self.threshold:
                #                 break
                # self.rel_r = self.abs_r - idx

                # row_len = self.abs_r - self.abs_l

                # ndtr((self.pt_partition[[0, -1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                # self.sink_tmp[0] = 1 - self.cdf_tmp[1] + self.cdf_tmp[0]

                # ndtr((self.pt_partition[[0, -1]] - x_i[1]) / self.b, out = self.cdf_tmp)
                # self.sink_tmp[1] = 1 - self.cdf_tmp[1] + self.cdf_tmp[0]
                # #
                # # ndtr((self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[0]) / self.b, out=self.cdf[:row_len+1])
                # # self.idx_tmp[0] = self.cdf[self.rel_l + 1] - self.cdf[self.rel_l]
                # # self.itvl_min_max[1][:self.rel_l] = self.cdf[1:self.rel_l+1] - self.cdf[:self.rel_l]
                # # self.itvl_min_max[0][self.rel_l+1:row_len] = self.cdf[self.rel_l+2:row_len+1] - self.cdf[self.rel_l+1:row_len]
                
                # # ndtr((self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[1]) / self.b, out=self.cdf[:row_len+1])
                # # self.idx_tmp[1] = self.cdf[self.rel_l + 1] - self.cdf[self.rel_l]
                # # self.itvl_min_max[0][:self.rel_l] = self.cdf[1:self.rel_l+1] - self.cdf[:self.rel_l]
                # # self.itvl_min_max[1][self.rel_l+1:row_len] = self.cdf[self.rel_l+2:row_len+1] - self.cdf[self.rel_l+1:row_len]
                # #

                # ndtr(-np.abs(self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[0]) / self.b, out=self.cdf[:row_len+1])
                # self.idx_tmp[0] = ndtr((x_i[1]-x_i[0])/self.b) - 0.5
                # self.itvl_min_max[1][:self.rel_l] = self.cdf[1:self.rel_l+1] - self.cdf[:self.rel_l]
                # self.itvl_min_max[0][self.rel_l+1:row_len] = -self.cdf[self.rel_l+2:row_len+1] + self.cdf[self.rel_l+1:row_len]
                
                # ndtr(-np.abs(self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[1]) / self.b, out=self.cdf[:row_len+1])
                # self.idx_tmp[1] = 0.5 - ndtr((x_i[0]-x_i[1])/self.b)
                # self.itvl_min_max[0][:self.rel_l] = self.cdf[1:self.rel_l+1] - self.cdf[:self.rel_l]
                # self.itvl_min_max[1][self.rel_l+1:row_len] = -self.cdf[self.rel_l+2:row_len+1] + self.cdf[self.rel_l+1:row_len]
                

                # #
                # self.itvl_min_max[0][self.rel_l] = min(self.idx_tmp)
                # self.itvl_min_max[1][self.rel_l] = 1 - 2 * ndtr(-(x_i[1] - x_i[0]) * 0.5 / self.b)

                # self.itvl_min_max[1][row_len] = max(self.sink_tmp)
                # if self.itvl_min_max[1][row_len] > self.threshold:
                #     if self.mid <= x_i[0] or self.mid >= x_i[1]:
                #         self.itvl_min_max[0][row_len] = min(self.sink_tmp)    
                #     else:
                #         self.itvl_min_max[0][row_len] = 2 * ndtr(-self.wid * 0.5 / self.b)
                #     index = list(range(self.abs_l,self.abs_r + 1))
                #     index[-1] = self.n
                # else: # self.itvl_min_max[1][row_len] <= self.threshold
                #     index = list(range(self.abs_l,self.abs_r))
                # return index
                # # binary search version ends version 2

                # binary search version starts version 3
                # determine abs_l and rel_l
                self.abs_l = idx - self.rel_l
                ndtr((self.pt_partition[[self.abs_l, self.abs_l + 1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                if self.cdf_tmp[1] * self.ndtr_rel_u - self.cdf_tmp[0] * self.ndtr_rel_l > self.threshold: # move left
                    while self.abs_l - 1 >= 0:
                        ndtr((self.pt_partition[[self.abs_l - 1, self.abs_l]] - x_i[0]) / self.b, out = self.cdf_tmp)
                        if self.cdf_tmp[1] * self.ndtr_rel_u - self.cdf_tmp[0] * self.ndtr_rel_l > self.threshold:
                            self.abs_l -= 1
                            # self.count += 1
                        else:
                            break
                else: # self.cdf_tmp[1] - self.cdf_tmp[0] <= self.threshold: move right
                    while 1 :
                        self.abs_l += 1
                        # self.count += 1
                        ndtr((self.pt_partition[[self.abs_l, self.abs_l + 1]] - x_i[0]) / self.b, out = self.cdf_tmp)
                        if self.cdf_tmp[1] * self.ndtr_rel_u - self.cdf_tmp[0] * self.ndtr_rel_l > self.threshold:
                            break
                self.rel_l = idx - self.abs_l
                # determine abs_r and rel_r
                self.abs_r = idx + self.rel_r
                if self.abs_r > self.n:
                    self.abs_r -= 1
                else:
                    ndtr(-(self.pt_partition[[self.abs_r - 1, self.abs_r]] - x_i[1]) / self.b, out = self.cdf_tmp)
                    if -(self.cdf_tmp[1] * self.ndtr_rel_u - self.cdf_tmp[0] * self.ndtr_rel_l) > self.threshold: # move right
                        while self.abs_r + 1 <= self.n:
                            ndtr(-(self.pt_partition[[self.abs_r, self.abs_r + 1]] - x_i[1]) / self.b, out = self.cdf_tmp)
                            if -(self.cdf_tmp[1] * self.ndtr_rel_u - self.cdf_tmp[0] * self.ndtr_rel_l) > self.threshold:
                                self.abs_r += 1
                                # self.count += 1
                            else:
                                break
                    else: # self.cdf_tmp[1] - self.cdf_tmp[0] <= self.threshold: move left
                        while 1 :
                            self.abs_r -= 1
                            # self.count += 1
                            ndtr(-(self.pt_partition[[self.abs_r - 1, self.abs_r]] - x_i[1]) / self.b, out = self.cdf_tmp)
                            if -(self.cdf_tmp[1] * self.ndtr_rel_u - self.cdf_tmp[0] * self.ndtr_rel_l) > self.threshold:
                                break
                self.rel_r = self.abs_r - idx

                row_len = self.abs_r - self.abs_l

                #
                # ndtr((self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[0]) / self.b, out=self.cdf[:row_len+1])
                # self.idx_tmp[0] = self.cdf[self.rel_l + 1] - self.cdf[self.rel_l]
                # self.itvl_min_max[1][:self.rel_l] = self.cdf[1:self.rel_l+1] - self.cdf[:self.rel_l]
                # self.itvl_min_max[0][self.rel_l+1:row_len] = self.cdf[self.rel_l+2:row_len+1] - self.cdf[self.rel_l+1:row_len]
                
                # ndtr((self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[1]) / self.b, out=self.cdf[:row_len+1])
                # self.idx_tmp[1] = self.cdf[self.rel_l + 1] - self.cdf[self.rel_l]
                # self.itvl_min_max[0][:self.rel_l] = self.cdf[1:self.rel_l+1] - self.cdf[:self.rel_l]
                # self.itvl_min_max[1][self.rel_l+1:row_len] = self.cdf[self.rel_l+2:row_len+1] - self.cdf[self.rel_l+1:row_len]
                #

                ndtr(-np.abs(self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[0]) / self.b, out=self.cdf[:row_len+1])
                # print(f'{self.cdf[row_len]}')
                
                # self.sink_tmp[0] = self.cdf[0] + self.cdf[row_len]
                
                self.target_min_max[1] = self.cdf[0] * self.ndtr_rel_u
                self.sink_min_max[0] = self.cdf[row_len] * self.ndtr_rel_l
                
                self.idx_tmp[0] = ndtr(wid_i/self.b) * self.ndtr_rel_l - 0.5
                self.itvl_min_max[1][:self.rel_l] = self.cdf[1:self.rel_l+1] * self.ndtr_rel_u - self.cdf[:self.rel_l] * self.ndtr_rel_l
                self.itvl_min_max[0][self.rel_l+1:row_len] = -self.cdf[self.rel_l+2:row_len+1] * self.ndtr_rel_l + self.cdf[self.rel_l+1:row_len] * self.ndtr_rel_u
                
                ndtr(-np.abs(self.pt_partition[self.abs_l : self.abs_r + 1] - x_i[1]) / self.b, out=self.cdf[:row_len+1])
                # print(f'{self.cdf[row_len]}')
                
                # self.sink_tmp[1] = self.cdf[0] + self.cdf[row_len]
                
                self.target_min_max[0] = self.cdf[0] * self.ndtr_rel_l
                self.sink_min_max[1] = self.cdf[row_len] * self.ndtr_rel_u
                
                self.idx_tmp[1] = 0.5 - ndtr(-wid_i/self.b) * self.ndtr_rel_u
                self.itvl_min_max[0][:self.rel_l] = self.cdf[1:self.rel_l+1] * self.ndtr_rel_l - self.cdf[:self.rel_l] * self.ndtr_rel_u
                self.itvl_min_max[1][self.rel_l+1:row_len] = -self.cdf[self.rel_l+2:row_len+1] * self.ndtr_rel_u + self.cdf[self.rel_l+1:row_len] * self.ndtr_rel_l

                #
                self.itvl_min_max[0][self.rel_l] = min(self.idx_tmp)
                self.itvl_min_max[1][self.rel_l] = 1 - 2 * ndtr(-wid_i * 0.5 / self.b) * self.ndtr_rel_l

                if self.target_min_max[1] > self.threshold:
                    self.tag_target_sink = 1
                else:
                    self.tag_target_sink = 0 
                if self.sink_min_max[1] > self.threshold:
                    self.tag_target_sink |= 2
                match self.tag_target_sink:
                    case 0:
                        index = list(range(self.abs_l,self.abs_r))
                    case 1:
                        self.itvl_min_max[:,row_len] = self.target_min_max
                        index = list(range(self.abs_l,self.abs_r + 1))
                        index[-1] = self.n
                    case 2:
                        self.itvl_min_max[:,row_len] = self.sink_min_max
                        index = list(range(self.abs_l,self.abs_r + 1))
                        index[-1] = self.n + 1
                    case 3:
                        self.itvl_min_max[:,row_len] = self.target_min_max
                        self.itvl_min_max[:,row_len+1] = self.sink_min_max
                        index = list(range(self.abs_l,self.abs_r + 2))
                        index[-2] = self.n
                        index[-1] = self.n + 1
                return index
                    
                # self.itvl_min_max[1][row_len] = max(self.sink_tmp)
                # if self.itvl_min_max[1][row_len] > self.threshold:
                #     self.count += 1
                #     if self.mid <= x_i[0] or self.mid >= x_i[1]:
                #         self.itvl_min_max[0][row_len] = min(self.sink_tmp)    
                #     else:
                #         self.itvl_min_max[0][row_len] = 2 * ndtr((self.pt_partition[self.abs_l] - self.mid) / self.b)
                #     index = list(range(self.abs_l,self.abs_r + 1))
                #     index[-1] = self.n
                # else: # self.itvl_min_max[1][row_len] <= self.threshold
                #     index = list(range(self.abs_l,self.abs_r))
                # return index
                # binary search version ends version 3

        else: # self.dim > 1
            pass

    def compute_label(self, targets):
        labels =[]
        for target in targets:
            labels.append(np.argwhere((self.pt_partition >= target[0]) & (self.pt_partition <= target[1])).squeeze())
        return labels




    # def output(self, idx):
    #     self.evaluate_itvl_prob(idx)
        
    #     index = np.argwhere(self.itvl_min_max[1]).squeeze()
    #     upper_bounds = self.itvl_min_max[1][index]
    #     lower_bounds = self.itvl_min_max[0][index]
    #     return index, lower_bounds, upper_bounds



# dim = 1
# dependency = 1
# symmetry = 1
# bound = 1e-3
# x = [-bound, bound]
# # x = [-1, 1]
# f = lambda x: x
# tag_f = 1
# cif_f = 0

# dt = 1e-10
# b_const = math.sqrt(dt)
# b = b_const
# tag_b = 1
# cif_b = 0
# precision = 1e-5/2

# imc = IMC(dim, dependency, symmetry, x, f, tag_f, cif_f, b, tag_b, cif_b, precision)

# index = imc.evaluate_itvl_prob(imc.n>>1)
# len = len(index)
# sum_l = np.sum(imc.itvl_min_max[0][:len])
# sum_u = np.sum(imc.itvl_min_max[1][:len])
# print(f'# nonzero: {len}\n{index}\n{imc.itvl_min_max[0][:len]}\n{imc.itvl_min_max[1][:len]}')
# err = (0.5 - ndtr(-imc.eta / imc.b)) + (1 - 2 * ndtr(-0.5 * imc.eta / imc.b))
# print(f'sum_l = {sum_l}, sum_u = {sum_u}, sum_u - sum_l = {sum_u - sum_l}, '\
#      + f'err = {err}, (sum_u - sum_l) - err = {sum_u - sum_l - err}')

# print(imc)

# a = 1 - 2 * ndtr(-(imc.pt_partition[1:]-imc.pt_partition[:-1]) * 0.5 / imc.b)
# b = 2 * ndtr((imc.pt_partition[1:]-imc.pt_partition[:-1]) * 0.5 / imc.b) - 1
# idx = np.argwhere((a-b)<0).squeeze()
# print(f'nonzero len = {idx.shape[0]}, {(a-b)[idx]}, min = {np.min((a-b)[idx])}, max = {np.max((a-b)[idx])}')

# print(f'a = {a}, b = {b}, a - b = {a-b}')

# index, lower_bounds, upper_bounds = imc.output(imc.n>>1)

# print(f'{index.shape[0]}\n{index}\n{lower_bounds}\n{upper_bounds}')

# sum_l = np.sum(lower_bounds)
# sum_u = np.sum(upper_bounds)
# err = (ndtr(imc.eta / imc.b) - 0.5) + (2 * ndtr(0.5 * imc.eta / imc.b) - 1)
# print(f'sum_l = {sum_l}, sum_u = {sum_u}, sum_u - sum_l = {sum_u - sum_l}, '\
#      + f'err = {err}, (sum_u - sum_l) - err = {sum_u - sum_l - err}')


# print(imc)



# dim = 1
# dependency = 1
# symmetry = 2
# x = [-1, 1]
# precision = 1e-4
# imc1 = IMC(dim, dependency, 2, x, f = 1, tag_f = 1, cif_f = 1, b = 1, tag_b = 1, cif_b = 1, precision = 1e-4)
# print(imc1.pt_partition, imc1.pt_partition.shape[0])
# imc2 = IMC(dim, dependency, 0, x, f = 1, tag_f = 1, cif_f = 1, b = 1, tag_b = 1, cif_b = 1, precision = 1)
# tmp = np.argwhere(imc1.pt_partition - imc2.pt_partition)
# print(tmp.shape[0])


# n = 10001
# tmp = (n >> 1) + 1
# wid = 2
# mid = 1
# pt_partition = np.zeros(n + 1)
# eta = wid / n
# pt_partition[tmp:] = np.array(range(tmp)) * eta + (eta / 2)
# pt_partition[:tmp] = np.flip(-pt_partition[tmp:])
# pt_partition += mid
# print(pt_partition)


# n = 3
# tmp = n >> 1
# wid = 2
# mid = 0
# print(np.array(range(-tmp,tmp + 1)) * (wid / n))



# x1 = Interval([3.0,3.2])
# x2 = Interval([2,5])


# z = Interval.cos(3)

# z = x1 + x2

# print(z)


# f = [[0]*3]*3
# f[0][0] =

# f_00 = 1
# f_01 = 2
# f_10 = lambda x: x+1
# f_11 = lambda x: x**2
# f = [[f_00,f_01], [f_10,f_11]]
