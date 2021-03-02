# cython: language_level=3

import numpy as np
cimport numpy as np

# import _catboost
# cimport _catboost

cdef class LoglossObjective_cython(object):
    cpdef float calc_ders_range(
            self,
            # _catboost._DoubleArrayWrapper approxes,
            np.ndarray[np.float64_t, ndim=1, mode='c'] approxes,
            np.ndarray[np.float64_t, ndim=1, mode='c']  targets,
            np.ndarray[np.float64_t, ndim=1, mode='c']  weights):

        cdef np.ndarray[np.float64_t, ndim=1] e, p, der1, der2
        cdef np.float64_t result

        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        e = np.exp(approxes)
        p = e / (1 + e)
        der1 = targets - p
        der2 = -p * (1 - p)

        if weights is not None:
            der1 *= weights
            der2 *= weights


        result = 0.5 # list(zip(der1, der2))
        return result




cpdef int test(int x):
    cdef int y = 1
    cdef int i
    for i in range(1, x+1):
        y *= i
    return y

