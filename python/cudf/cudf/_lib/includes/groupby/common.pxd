# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf import *
from cudf._lib.cudf cimport *

cdef extern from "cudf/groupby.hpp" namespace "cudf::groupby" nogil:

    ctypedef enum operators:
        SUM,
        MIN,
        MAX,
        COUNT,
        MEAN,
        MEDIAN,
        QUANTILE

    cdef cppclass Options:
        Options(bool _ignore_null_keys) except +
        Options() except +
        const bool ignore_null_keys
