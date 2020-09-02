# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type, null_equality

cdef extern from "cudf/join.hpp" namespace "cudf" nogil:

    cdef unique_ptr[table] inner_join(
        const table_view left,
        const table_view right,
        const vector[size_type] left_on,
        const vector[size_type] right_on,
        const vector[pair[size_type, size_type]] columns_in_common
    ) except +

    cdef unique_ptr[table] left_join(
        const table_view left,
        const table_view right,
        const vector[size_type] left_on,
        const vector[size_type] right_on,
        const vector[pair[size_type, size_type]] columns_in_common
    ) except +

    cdef unique_ptr[table] full_join(
        const table_view left,
        const table_view right,
        const vector[size_type] left_on,
        const vector[size_type] right_on,
        const vector[pair[size_type, size_type]] columns_in_common
    ) except +

    cdef unique_ptr[table] left_semi_join(
        const table_view left,
        const table_view right,
        const vector[size_type] left_on,
        const vector[size_type] right_on,
        const vector[size_type] return_columns
    ) except +

    cdef unique_ptr[table] left_anti_join(
        const table_view left,
        const table_view right,
        const vector[size_type] left_on,
        const vector[size_type] right_on,
        const vector[size_type] return_columns
    ) except +

    enum common_columns_output_side:
        PROBE "cudf::hash_join:common_columns_output_side::PROBE"
        BUILD "cudf::hash_join::common_columns_output_side::BUILD"

    cdef cppclass hash_join:
        hash_join(const table_view& build, vector[size_type] build_on) except +

        pair[unique_ptr[table], unique_ptr[table]] inner_join(
            const table_view& probe,
            const vector[size_type]& probe_on,
            const vector[pair[size_type, size_type]]& columns_in_common,
        ) except +

        pair[unique_ptr[table], unique_ptr[table]] inner_join(
            const table_view& probe,
            const vector[size_type]& probe_on,
            const vector[pair[size_type, size_type]]& columns_in_common,
            common_columns_output_side
        ) except +

        pair[unique_ptr[table], unique_ptr[table]] inner_join(
            const table_view& probe,
            const vector[size_type]& probe_on,
            const vector[pair[size_type, size_type]]& columns_in_common,
            common_columns_output_side,
            null_equality
        ) except +

        pair[unique_ptr[table], unique_ptr[table]] left_join(
            const table_view& probe,
            const vector[size_type]& probe_on,
            const vector[pair[size_type, size_type]]& columns_in_common,
        ) except +

        pair[unique_ptr[table], unique_ptr[table]] left_join(
            const table_view& probe,
            const vector[size_type]& probe_on,
            const vector[pair[size_type, size_type]]& columns_in_common,
            null_equality
        ) except +

        pair[unique_ptr[table], unique_ptr[table]] full_join(
            const table_view& probe,
            const vector[size_type]& probe_on,
            const vector[pair[size_type, size_type]]& columns_in_common,
        ) except +

        pair[unique_ptr[table], unique_ptr[table]] full_join(
            const table_view& probe,
            const vector[size_type]& probe_on,
            const vector[pair[size_type, size_type]]& columns_in_common,
            null_equality
        ) except +


cdef class HashJoin:
    cdef hash_join c_obj
    cdef Table build_table
