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

    result_col_names = compute_result_col_names(lhs, rhs, how)

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

    all_cols_py = columns_from_ptr(move(c_result))
    data_ordered_dict = OrderedDict(zip(result_col_names, all_cols_py))
    return Table(data=data_ordered_dict, index=None)


def compute_result_col_names(lhs, rhs, how):
    """
    Determine the names of the data columns in the result of
    a libcudf join, based on the original left and right frames
    as well as the type of join that was performed.
    """
    if how in {"left", "inner", "outer", "leftsemi", "leftanti"}:
        a = lhs._data.keys()
        if how not in {"leftsemi", "leftanti"}:
            return list(chain(a, (k for k in rhs._data.keys()
                        if k not in lhs._data.keys())))
        return list(a)
    else:
        raise NotImplementedError(
            f"{how} merge not supported yet"
        )
