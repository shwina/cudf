# Copyright (c) 2020, NVIDIA CORPORATION.

from collections import OrderedDict
from itertools import chain

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._lib.table cimport Table, columns_from_ptr
from cudf._lib.move cimport move

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.join as cpp_join


cpdef join(
    Table lhs,
    Table rhs,
    object how,
    vector[int] left_on_ind,
    vector[int] right_on_ind,
    vector[pair[int, int]] columns_in_common,
    object result_col_names
):
    """
    Call libcudf++ join for full outer, inner and left joins.
    """
    # Views might or might not include index
    cdef table_view lhs_view = lhs.data_view()
    cdef table_view rhs_view = rhs.data_view()

    # Only used for semi or anti joins
    # The result columns are only the left hand columns
    cdef vector[int] all_left_inds = range(
        lhs._num_columns
    )
    cdef vector[int] all_right_inds = range(
        rhs._num_columns
    )

    cdef unique_ptr[table] c_result
    if how == 'inner':
        with nogil:
            c_result = move(cpp_join.inner_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
    elif how == 'left':
        with nogil:
            c_result = move(cpp_join.left_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
    elif how == 'outer':
        with nogil:
            c_result = move(cpp_join.full_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
    elif how == 'leftsemi':
        with nogil:
            c_result = move(cpp_join.left_semi_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                all_left_inds
            ))
    elif how == 'leftanti':
        with nogil:
            c_result = move(cpp_join.left_anti_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                all_left_inds
            ))

    return Table.from_unique_ptr(
        move(c_result), column_names=result_col_names
    )
