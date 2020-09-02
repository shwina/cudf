# Copyright (c) 2020, NVIDIA CORPORATION.
import itertools
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf.core.dtypes import CategoricalDtype


class Merge(object):
    def __init__(
        self,
        lhs,
        rhs,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        sort,
        lsuffix,
        rsuffix,
        method,
        indicator,
        suffixes,
    ):
        """
        Manage the merging of two Frames.

        Parameters
        ----------
        lhs : Series or DataFrame
            The left operand of the merge
        rhs : Series or DataFrame
            The right operand of the merge
        on : string or list like
            A set of key columns in the left and right operands
            elements must be common to both frames
        left_on : string or list like
            A set of key columns in the left operand. Must be
            specified with right_on or right_index concurrently
        right_on : string or list like
            A set of key columns in the right operand. Must be
            specified with left_on or left_index concurrently
        left_index : bool
            Boolean flag indicating the left index column or columns
            are to be used as join keys in order.
        right_index : bool
            Boolean flag indicating the right index column or coumns
            are to be used as join keys in order.
        how : string
            The type of join. Possible values are
            'inner', 'outer', 'left', 'leftsemi' and 'leftanti'
        sort : bool
            Boolean flag indicating if the output Frame is to be
            sorted on the output's join keys, in left to right order.
        lsuffix : string
            The suffix to be appended to left hand column names that
            are found to exist in the right frame, but are not specified
            as join keys themselves.
        rsuffix : string
            The suffix to be appended to right hand column names that
            are found to exist in the left frame, but are not specified
            as join keys themselves.
        suffixes : list like
            Left and right suffixes specified together, unpacked into lsuffix
            and rsuffix.
        """
        self.lhs = lhs
        self.rhs = rhs
        self.left_index = left_index
        self.right_index = right_index
        self.method = method
        self.sort = sort

        # check that the merge is valid

        self.validate_merge_cfg(
            lhs,
            rhs,
            on,
            left_on,
            right_on,
            left_index,
            right_index,
            how,
            lsuffix,
            rsuffix,
            suffixes,
        )
        self.how = how
        self.preprocess_merge_params(
            on, left_on, right_on, lsuffix, rsuffix, suffixes
        )

    def perform_merge(self):
        """
        Call libcudf to perform a merge between the operands. If
        necessary, cast the input key columns to compatible types.
        Potentially also cast the output back to categorical.
        """
        result_col_order = compute_result_col_names(
            self.lhs, self.rhs, self.how
        )

        (
            left_on_inds,
            right_on_inds,
            columns_in_common,
            all_left_inds,
            result_idx_positions,
            result_col_names,
            result_index_names,
        ) = self.compute_join_indices()

        if self.left_index or self.right_index:
            self.lhs = self.lhs.reset_index()
            self.rhs = self.rhs.reset_index()

        output_dtypes = self.compute_output_dtypes(columns_in_common)
        self.typecast_input_to_libcudf(left_on_inds, right_on_inds)
        result = libcudf.join.join(
            self.lhs,
            self.rhs,
            self.how,
            self.method,
            left_on_inds,
            right_on_inds,
            columns_in_common,
            all_left_inds,
        )
        result = self.typecast_libcudf_to_output(result, output_dtypes)

        if self.left_index or self.right_index:
            ind_cols = OrderedDict()
            for name, pos in zip(
                result_index_names[::-1], result_idx_positions[::-1]
            ):
                ind_cols[name] = result._data.pop(pos)
            index = OrderedDict()
            for k, v in reversed(ind_cols.items()):
                index[k] = v
            index = cudf.core.frame.Frame(index)
        else:
            index = None

        data = {
            key: value
            for key, value in zip(result_col_names, result._data.values())
        }

        result = cudf.core.frame.Frame(data=data, index=index)
        result = self.out_class._from_table(result)

        if isinstance(result, cudf.Index):
            return result
        else:
            return result[result_col_order]

    def preprocess_merge_params(
        self, on, left_on, right_on, lsuffix, rsuffix, suffixes
    ):
        """
        Translate a valid configuration of user input parameters into
        the subset of input configurations handled by the cython layer.
        Apply suffixes to columns.
        """

        self.out_class = cudf.DataFrame
        if isinstance(self.lhs, cudf.MultiIndex) or isinstance(
            self.rhs, cudf.MultiIndex
        ):
            self.out_class = cudf.MultiIndex
        elif isinstance(self.lhs, cudf.Index):
            self.out_class = self.lhs.__class__

        if on:
            on = [on] if isinstance(on, str) else list(on)
            left_on = right_on = on
        else:
            if left_on:
                left_on = (
                    [left_on] if isinstance(left_on, str) else list(left_on)
                )
            if right_on:
                right_on = (
                    [right_on] if isinstance(right_on, str) else list(right_on)
                )

        same_named_columns = set(self.lhs._data.keys()) & set(
            self.rhs._data.keys()
        )
        if not (left_on or right_on) and not (
            self.left_index and self.right_index
        ):
            left_on = right_on = list(same_named_columns)

        no_suffix_cols = []
        if left_on and right_on:
            no_suffix_cols = [
                left_name
                for left_name, right_name in zip(left_on, right_on)
                if left_name == right_name and left_name in same_named_columns
            ]

        if suffixes:
            lsuffix, rsuffix = suffixes
        for name in same_named_columns:
            if name not in no_suffix_cols:
                self.lhs.rename(
                    {name: f"{name}{lsuffix}"}, inplace=True, axis=1
                )
                self.rhs.rename(
                    {name: f"{name}{rsuffix}"}, inplace=True, axis=1
                )
                if left_on and name in left_on:
                    left_on[left_on.index(name)] = f"{name}{lsuffix}"
                if right_on and name in right_on:
                    right_on[right_on.index(name)] = f"{name}{rsuffix}"

        self.left_on = left_on if left_on is not None else []
        self.right_on = right_on if right_on is not None else []
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix

    @staticmethod
    def validate_merge_cfg(
        lhs,
        rhs,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        lsuffix,
        rsuffix,
        suffixes,
    ):
        """
        Error for various invalid combinations of merge input parameters
        """

        # must actually support the requested merge type
        if how not in {"left", "inner", "outer", "leftanti", "leftsemi"}:
            raise NotImplementedError(f"{how} merge not supported yet")

        # Passing 'on' with 'left_on' or 'right_on' is ambiguous
        if on and (left_on or right_on):
            raise ValueError(
                'Can only pass argument "on" OR "left_on" '
                'and "right_on", not a combination of both.'
            )

        # Can't merge on unnamed Series
        if (isinstance(lhs, cudf.Series) and not lhs.name) or (
            isinstance(rhs, cudf.Series) and not rhs.name
        ):
            raise ValueError("Can not merge on unnamed Series")

        # Keys need to be in their corresponding operands
        if on:
            if isinstance(on, str):
                on_keys = [on]
            elif isinstance(on, tuple):
                on_keys = list(on)
            else:
                on_keys = on
            for key in on_keys:
                if not (key in lhs._data.keys() and key in rhs._data.keys()):
                    raise KeyError(f"on key {on} not in both operands")
        elif left_on and right_on:
            left_on_keys = (
                [left_on] if not isinstance(left_on, list) else left_on
            )
            right_on_keys = (
                [right_on] if not isinstance(right_on, list) else right_on
            )

            for key in left_on_keys:
                if key not in lhs._data.keys():
                    raise KeyError(f'Key "{key}" not in left operand')
            for key in right_on_keys:
                if key not in rhs._data.keys():
                    raise KeyError(f'Key "{key}" not in right operand')

        # Require same total number of columns to join on in both operands
        len_left_on = 0
        len_right_on = 0
        if left_on:
            len_left_on += (
                len(left_on) if pd.api.types.is_list_like(left_on) else 1
            )
        if right_on:
            len_right_on += (
                len(right_on) if pd.api.types.is_list_like(right_on) else 1
            )
        if not (len_left_on + left_index * lhs._num_indices) == (
            len_right_on + right_index * rhs._num_indices
        ):
            raise ValueError(
                "Merge operands must have same number of join key columns"
            )

        # If nothing specified, must have common cols to use implicitly
        same_named_columns = set(lhs._data.keys()) & set(rhs._data.keys())
        if (
            not (left_index or right_index)
            and not (left_on or right_on)
            and len(same_named_columns) == 0
        ):
            raise ValueError("No common columns to perform merge on")

        if suffixes:
            lsuffix, rsuffix = suffixes
        for name in same_named_columns:
            if name == left_on == right_on:
                continue
            elif left_on and right_on:
                if (name in left_on and name in right_on) and (
                    left_on.index(name) == right_on.index(name)
                ):
                    continue
            else:
                if not (lsuffix or rsuffix):
                    raise ValueError(
                        "there are overlapping columns but "
                        "lsuffix and rsuffix are not defined"
                    )

    def typecast_input_to_libcudf(self, left_on_ind, right_on_ind):
        """
        Check each pair of join keys in the left and right hand
        operands and apply casting rules to match their types
        before passing the result to libcudf.
        """
        left_columns = self.lhs._data.columns
        right_columns = self.rhs._data.columns

        for (left_on_index, right_on_index) in zip(left_on_ind, right_on_ind):
            to_dtype = self.input_to_libcudf_casting_rules(
                left_columns[left_on_index],
                right_columns[right_on_index],
                self.how,
            )
            self.lhs._data[self.lhs._data.names[left_on_index]] = left_columns[
                left_on_index
            ].astype(to_dtype)
            self.rhs._data[
                self.rhs._data.names[right_on_index]
            ] = right_columns[right_on_index].astype(to_dtype)

    def input_to_libcudf_casting_rules(self, lcol, rcol, how):
        """
        Determine what dtype the left and right hand
        input columns must be cast to for a libcudf
        join to proceed.
        """

        cast_warn = (
            "can't safely cast column from {} with type"
            " {} to {}, upcasting to {}"
        )
        ctgry_err = (
            "can't implicitly cast column {0} to categories"
            " from {1} during {1} join"
        )

        dtype_l = lcol.dtype
        dtype_r = rcol.dtype
        libcudf_join_type = None
        if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
            # if categorical and equal, children passed to libcudf
            libcudf_join_type = dtype_l
        elif isinstance(dtype_l, CategoricalDtype) and isinstance(
            dtype_r, CategoricalDtype
        ):
            # categories are not equal
            libcudf_join_type = np.dtype("O")
        elif how == "left":
            check_col = rcol.fillna(0)
            if not check_col.can_cast_safely(dtype_l):
                libcudf_join_type = self.input_to_libcudf_casting_rules(
                    lcol, rcol, "inner"
                )
                warnings.warn(
                    cast_warn.format(
                        "right", dtype_r, dtype_l, libcudf_join_type
                    )
                )
            else:
                libcudf_join_type = dtype_l
        elif how == "right":
            check_col = lcol.fillna(0)
            if not check_col.can_cast_safely(dtype_r):
                libcudf_join_type = self.input_to_libcudf_casting_rules(
                    lcol, rcol, "inner"
                )
                warnings.warn(
                    cast_warn.format(
                        "left", dtype_l, dtype_r, libcudf_join_type
                    )
                )
            else:
                libcudf_join_type = dtype_r

        elif isinstance(dtype_l, CategoricalDtype):
            if how == "right":
                raise ValueError(ctgry_err.format(rcol, "right"))
            libcudf_join_type = lcol.cat().categories.dtype
        elif isinstance(dtype_r, CategoricalDtype):
            if how == "left":
                raise ValueError(ctgry_err.format(lcol, "left"))
            libcudf_join_type = rcol.cat().categories.dtype
        elif how in {"inner", "outer"}:
            if (np.issubdtype(dtype_l, np.number)) and (
                np.issubdtype(dtype_r, np.number)
            ):
                if dtype_l.kind == dtype_r.kind:
                    # both ints or both floats
                    libcudf_join_type = max(dtype_l, dtype_r)
                else:
                    libcudf_join_type = np.find_common_type(
                        [], [dtype_l, dtype_r]
                    )
            elif np.issubdtype(dtype_l, np.datetime64) and np.issubdtype(
                dtype_r, np.datetime64
            ):
                libcudf_join_type = max(dtype_l, dtype_r)
        return libcudf_join_type

    def libcudf_to_output_casting_rules(self, lcol, rcol, how):
        """
        Determine what dtype an output merge key column should be
        cast to after it has been processed by libcudf. Determine
        if a column should be promoted to a categorical datatype.
        """
        dtype_l = lcol.dtype
        dtype_r = rcol.dtype

        merge_return_type = None
        # we  currently only need to do this for categorical variables
        if isinstance(dtype_l, CategoricalDtype) and isinstance(
            dtype_r, CategoricalDtype
        ):
            if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
                if how in {"inner", "left"}:
                    merge_return_type = dtype_l
                elif how == "outer" and not (
                    dtype_l.ordered or dtype_r.ordered
                ):
                    new_cats = cudf.concat(
                        dtype_l.categories, dtype_r.categories
                    ).unique()
                    merge_return_type = cudf.core.dtypes.CategoricalDtype(
                        categories=new_cats
                    )
            else:
                merge_return_type = "category"
        return merge_return_type

    def compute_output_dtypes(self, columns_in_common):
        """
        Determine what datatypes should be applied to the result
        of a libcudf join, baesd on the original left and right
        frames.
        """
        data_dtypes = []

        left_columns = list(self.lhs._data.columns)
        right_columns = list(self.rhs._data.columns)

        for drop_index in sorted(
            (x[1] for x in columns_in_common), reverse=True
        ):
            del right_columns[drop_index]
        all_columns = left_columns + right_columns

        for col in all_columns:
            data_dtypes.append(col.dtype)

        for (left_on_index, right_on_index) in columns_in_common:
            data_dtypes[left_on_index] = self.libcudf_to_output_casting_rules(
                self.lhs._data.columns[left_on_index],
                self.rhs._data.columns[right_on_index],
                self.how,
            )

        return data_dtypes

    def typecast_libcudf_to_output(self, output, output_dtypes):
        """
        Apply precomputed output index and data column data types
        to the output of a libcudf join.
        """
        data_dtypes = output_dtypes
        for data_col_lbl, data_col in output._data.items():
            data_dtype = data_dtypes[data_col_lbl]
            if data_dtype:
                output._data[data_col_lbl] = self._build_output_col(
                    data_col, data_dtype
                )
        return output

    def _build_output_col(self, col, dtype):

        if isinstance(
            dtype, (cudf.core.dtypes.CategoricalDtype, pd.CategoricalDtype)
        ):
            outcol = cudf.core.column.build_categorical_column(
                categories=dtype.categories,
                codes=col.set_mask(None),
                mask=col.base_mask,
                ordered=dtype.ordered,
            )
        else:
            outcol = col.astype(dtype)
        return outcol

    def compute_join_indices(self):
        left_on_ind = []
        right_on_ind = []
        columns_in_common = OrderedDict()
        result_idx_positions = []
        result_index_names = []

        result_col_names = compute_result_col_names(
            self.lhs, self.rhs, self.how
        )

        all_left_inds = range(
            self.lhs._num_columns + (self.lhs._num_indices * self.left_index)
        )

        # keep track of where the desired index column will end up
        if self.left_index or self.right_index:
            left_join_cols = list(self.lhs._index_names) + list(
                self.lhs._data.keys()
            )
            right_join_cols = list(self.rhs._index_names) + list(
                self.rhs._data.keys()
            )
            if self.left_index and self.right_index:
                # Index columns will be common, on the left, dropped from right
                # Index name is from the left
                # Both views, must take index column indices
                left_on_indices = right_on_indices = range(
                    self.lhs._num_indices
                )
                result_idx_positions = range(self.lhs._num_indices)
                result_index_names = self.lhs._index_names

            elif self.left_index:
                # Joins left index columns with right 'on' columns
                left_on_indices = range(self.lhs._num_indices)
                right_on_indices = [
                    right_join_cols.index(on_col) for on_col in self.right_on
                ]

                # The left index columns 'become' the new RHS columns
                # and the right index 'survives'
                result_idx_positions = range(
                    len(left_join_cols),
                    len(left_join_cols) + self.lhs._num_indices,
                )
                result_index_names = self.rhs._index_names

                # but since the common columns are gathered from the left
                # the rhs 'on' cols are returned on the left of the result
                # rearrange the names so account for this
                common = [None] * self.rhs._num_indices
                for i in range(self.rhs._num_indices):
                    common[i] = result_col_names.pop(
                        result_col_names.index(self.right_on[i])
                    )
                result_col_names = common + result_col_names
            elif self.right_index:
                # Joins right index columns with left 'on' columns
                right_on_indices = range(self.rhs._num_indices)
                left_on_indices = [
                    left_join_cols.index(on_col) for on_col in self.left_on
                ]

                # The right index columns 'become' the new LHS columns
                # and the left index survives
                # since they are already gathered from the left,
                # no rearranging has to be done
                result_idx_positions = range(self.lhs._num_indices)
                result_index_names = self.lhs._index_names
            for i_l, i_r in zip(left_on_indices, right_on_indices):
                left_on_ind.append(i_l)
                right_on_ind.append(i_r)
                columns_in_common[(i_l, i_r)] = None
        else:
            left_join_cols = list(self.lhs._data.keys())
            right_join_cols = list(self.rhs._data.keys())

        # If both left/right_index, joining on indices plus additional cols
        # If neither, joining on just cols, not indices
        # In both cases, must match up additional column indices in lhs/rhs
        if self.left_index == self.right_index:
            for name in self.left_on:
                left_on_ind.append(left_join_cols.index(name))
                if name in self.right_on:
                    if self.left_on.index(name) == self.right_on.index(name):
                        columns_in_common[
                            (
                                left_join_cols.index(name),
                                right_join_cols.index(name),
                            )
                        ] = None
            for name in self.right_on:
                right_on_ind.append(right_join_cols.index(name))

        columns_in_common = list(columns_in_common.keys())

        return (
            left_on_ind,
            right_on_ind,
            columns_in_common,
            all_left_inds,
            result_idx_positions,
            result_col_names,
            result_index_names,
        )


def compute_result_col_names(lhs, rhs, how):
    """
    Determine the names of the data columns in the result of
    a libcudf join, based on the original left and right frames
    as well as the type of join that was performed.
    """
    if how in {"left", "inner", "outer", "leftsemi", "leftanti"}:
        a = lhs._data.keys()
        if how not in {"leftsemi", "leftanti"}:
            return list(
                itertools.chain(
                    a,
                    (k for k in rhs._data.keys() if k not in lhs._data.keys()),
                )
            )
        return list(a)
    else:
        raise NotImplementedError(f"{how} merge not supported yet")
