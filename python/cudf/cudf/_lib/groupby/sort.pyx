from libcpp.memory cimport unique_ptr

cimport cudf._lib.includes.groupby.common as groupby_common
from cudf._lib.includes.groupby.sort cimport *
from cudf._lib.table cimport Table, TableView
from cudf._lib.utils cimport *
import cudf._lib.quantile as quantile

from cudf.utils.dtypes import is_scalar


cdef class GroupBy:
    cdef vector[operation] c_ops
    cdef pair[cudf_table, vector[gdf_column_ptr]] c_result
    cdef dict __dict__
    
    def __cinit__(self, df, by):
        pass
        
    def __init__(self, df, by):
        self.df = df
        if not isinstance(by, list):
            by = [by]
        self.keys = by
        self.values = [col for col in self.df._columns
                       if col not in self.keys]

    def quantile(self, q=0.5, interpolation='linear'):
        q = [q] if is_scalar(q) else list(q)
        cdef unique_ptr[operation_args] qargs = unique_ptr[operation_args](
            new quantile_args(q, quantile._QUANTILE_METHODS[interpolation])
        )
        self.c_ops.push_back(operation(groupby_common.QUANTILE, move(qargs)))

    def max(self):
        cdef unique_ptr[operation_args] args = unique_ptr[operation_args](new operation_args())
        self.c_ops.push_back(operation(groupby_common.MAX, move(args)))

    def min(self):
        cdef unique_ptr[operation_args] args = unique_ptr[operation_args](new operation_args())
        self.c_ops.push_back(operation(groupby_common.MIN, move(args)))
        
    def agg(self, func):
        if isinstance(func, list):
            assert len(self.values) == len(func)
            for f in func:
                f(self)
        else:
            raise NotImplementedError

        keys_table = TableView(self.keys)
        values_table = TableView(self.values)

        c_result = groupby(
            keys_table.ptr[0],
            values_table.ptr[0],
            self.c_ops)
        
        self.c_ops.clear()
        
        result_keys_table  = Table.from_ptr(&c_result.first)
        cdef cudf_table c_result_values_table = cudf_table(c_result.second)
        result_values_table = Table.from_ptr(&c_result_values_table, own=False)
        
        return result_keys_table.release(), result_values_table.release()
