# Copyright (c) 2020, NVIDIA CORPORATION.

from typing import Any, Mapping, Tuple


class CachedInstanceMeta(type):
    __instances: Mapping[Tuple, Any] = {}

    def __call__(self, *args, **kwargs):
        arg_tuple = args + tuple(kwargs.values())
        if arg_tuple in self.__instances:
            return self.__instances[arg_tuple]
        else:
            obj = super().__call__(*args, **kwargs)
            self.__instances[arg_tuple] = obj
            return obj
