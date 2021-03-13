# cython: language_level=3

# setup with: python setup.py build_ext --inplace

import numpy as np
cimport numpy as cnp
import catboost._catboost
from catboost._catboost import _FloatArrayWrapper, _DoubleArrayWrapper
# import _catboost
# cimport _catboost

cdef class LoglossObjective_cython(object):
    cpdef calc_ders_range(
            self,
            approx,
            target,
            weight):

        # cdef cnp.ndarray[cnp.float64_t, ndim=1] e, p, der1, der2

        cdef cnp.ndarray[cnp.float64_t, ndim=1] approxes = np.array([ap for ap in approx])
        # print(type(approxes), len(approxes), approxes.mean(), approxes.max())
        cdef cnp.ndarray[cnp.float64_t, ndim=1] targets = np.array([ap for ap in target])
        # print(type(targets), len(targets), targets.mean(), targets.max())

        if weight is None:
            weights = None
        else:
            weights = _FloatArrayWrapper.create(weight.data(), len(weight))

        # cdef int i
        # cdef double app, targ
        cdef cnp.ndarray[cnp.float64_t, ndim=1] e = np.exp(approxes)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] p = e / (1 + e)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] der1 = targets - p
        # print(type(der1), len(der1), np.mean(der1), np.mean(der1))
        
        cdef cnp.ndarray[cnp.float64_t, ndim=1] der2 = -p * (1 - p)
        # print(type(der2), len(der2), np.mean(der2), np.mean(der2)())

        # if weight is not None:
        #     for i, wei  in enumerate(weight):
        #         weights[i]  = wei
        #
        #     der1 *= weights
        #     der2 *= weights


        result = list(zip(der1, der2))
        # print(type(result))
        return result




cpdef int test(int x):
    cdef int y = 1
    cdef int i
    for i in range(1, x+1):
        y *= i
    return y

