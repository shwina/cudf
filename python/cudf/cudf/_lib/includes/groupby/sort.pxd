# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from cudf._lib.cudf import *
from cudf._lib.cudf cimport *
from cudf._lib.includes.quantile cimport interpolation

cimport cudf._lib.includes.groupby.common as groupby_common

cdef extern from "cudf/groupby.hpp" namespace "cudf::groupby::sort" nogil:
    cdef cppclass operation_args "cudf::groupby::sort::operation_args":
        pass

    cdef cppclass quantile_args "cudf::groupby::sort::quantile_args":
        quantile_args(vector[double] quantiles, interpolation interp) except +
        vector[double] quantiles
        interpolation interp

    cdef cppclass operation "cudf::groupby::sort::operation":
        groupby_common.operators op_name
        unique_ptr[operation_args] args

    cdef enum null_order "cudf::groupby::sort::null_order":
        AFTER "cudf::groupby::sort::null_order::AFTER"
        BEFORE "cudf::groupby::sort::null_order::BEFORE"


cdef extern from "cudf/groupby.hpp" nogil:
    cdef pair[cudf_table, gdf_column] gdf_group_by_without_aggregations(
        const cudf_table  cols,
        gdf_size_type num_key_cols,
        const gdf_index_type* key_col_indices,
        gdf_context* context
    ) except +
