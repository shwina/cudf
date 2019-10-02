from libcpp.memory cimport unique_ptr

cimport cudf._lib.includes.groupby.common as groupby_common
from cudf._lib.includes.groupby.sort cimport *
from cudf._lib.utils cimport *
import cudf._lib.quantile as quantile


cdef class GroupBy:

    cdef vector[operation] c_ops
    cdef pair[cudf_table, vector[gdf_column_ptr]] c_result
    cdef cudf_table *c_keys_table
    cdef cudf_table *c_values_table
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
        cdef unique_ptr[operation_args] qargs = unique_ptr[operation_args](
            new quantile_args(q, quantile._QUANTILE_METHODS[interpolation])
        )
        self.c_ops.push_back(operation(groupby_common.QUANTILE, move(qargs)))

    def max(self):
        cdef unique_ptr[operation_args] args = unique_ptr[operation_args](new operation_args())
        self.c_ops.push_back(operation(groupby_common.MAX, move(args)))
        
    def agg(self, func):
        if isinstance(func, list):
            assert len(self.values) == len(func)
            for f in func:
                f(self)
        else:
            raise NotImplementedError

        self.c_keys_table = table_from_columns(self.keys)
        self.c_values_table = table_from_columns(self.values)

        result = groupby(
            self.c_keys_table[0],
            self.c_values_table[0],
            self.c_ops)

        cdef cudf_table result_values_table = cudf_table(result.second)
        result_keys = columns_from_table(&result.first)
        result_values = columns_from_table(&result_values_table)
        return result_keys, result_values

    def __dealloc__(self):
        del self.c_keys_table
        del self.c_values_table
