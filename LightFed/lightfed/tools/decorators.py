from collections import defaultdict
from functools import wraps
import logging
import types
import time


def logging_time_cost(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        logging.debug(f"start func: {func.__name__}")
        t1 = time.time()
        ans = func(*args, **kwargs)
        t2 = time.time()
        logging.debug(f"finish func: {func.__name__}, time cost: {t2 - t1}s")
        return ans

    return decorated


# 用在类函数中
# 当类函数调用次数达到wait_times时才会真正执行
# class SyncFunc:
#     def __init__(self, func):
#         wraps(func)(self)
#         self.func = func
#         self.wait_times = -1
#
#     def __call__(self, *args, **kwargs):
#         self.wait_times -= 1
#         if not self.wait_times:
#             self.func(*args, **kwargs)
#
#     def set_wait_times(self, wait_times):
#         self.wait_times = wait_times
#
#     def __get__(self, instance, cls):
#         if instance is None:
#             return self
#         else:
#             return types.MethodType(self, instance)
#
#
# # 用在类函数中
# def group_params(condition_func):
#     def decorater(func):
#         _varnames_ = func.__code__.co_varnames[1:]
#         _defaults_ = func.__defaults__
#
#         _params_ = defaultdict(lambda: {varname: [] for varname in _varnames_})
#         _param_defaults_ = dict(zip(_varnames_[-len(_defaults_):], _defaults_))
#         _varnames_set_ = set(_varnames_)
#
#         @wraps(func)
#         def wrapper(_self, *args, **kwargs):
#             unknown_params = kwargs.keys() - _varnames_set_
#             if len(unknown_params):
#                 raise ValueError(
#                     f"function:{func.__name__}{_varnames_} got unexpected keyword arguments:{tuple(unknown_params)}")
#
#             local_params = _params_[id(_self)]
#
#             for i, varname in enumerate(_varnames_):
#                 if i < len(args):
#                     local_params[varname].append(args[i])
#                 elif varname in kwargs:
#                     local_params[varname].append(kwargs[varname])
#                 elif varname in _param_defaults_:
#                     local_params[varname].append(_param_defaults_[varname])
#                 else:
#                     raise ValueError(f"function:{func.__name__}{_varnames_} missing arguments:{varname}")
#
#             if condition_func(_self, local_params):
#                 func(_self, **local_params)
#                 for param_list in local_params.values():
#                     param_list.clear()
#
#         return wrapper
#
#     return decorater
#
#
# # 用在类函数中
# def reduce_params(reduce_func_dict, condition_func):
#     def decorater(func):
#         _varnames_ = func.__code__.co_varnames[1:]
#         _defaults_ = func.__defaults__
#
#         _params_ = defaultdict(lambda: {varname: None for varname in _varnames_})
#         _param_defaults_ = dict(zip(_varnames_[-len(_defaults_):], _defaults_))
#         _varnames_set_ = set(_varnames_)
#
#         if reduce_func_dict.keys() != _varnames_set_:
#             raise ValueError(
#                 f"function:{func.__name__}{_varnames_} doesn't match the reduce func:{tuple(reduce_func_dict.keys())}")
#
#         @wraps(func)
#         def wrapper(_self, *args, **kwargs):
#             unknown_params = kwargs.keys() - _varnames_set_
#             if len(unknown_params):
#                 raise ValueError(
#                     f"function:{func.__name__}{_varnames_} got unexpected keyword arguments:{tuple(unknown_params)}")
#
#             local_params = _params_[id(_self)]
#             for i, varname in enumerate(_varnames_):
#                 if i < len(args):
#                     varvalue2 = args[i]
#                 elif varname in kwargs:
#                     varvalue2 = kwargs[varname]
#                 elif varname in _param_defaults_:
#                     varvalue2 = _param_defaults_[varname]
#                 else:
#                     raise ValueError(f"function:{func.__name__}{_varnames_} missing arguments:{varname}")
#
#                 varvalue1 = local_params[varname]
#                 if varvalue1 is None:
#                     local_params[varname] = varvalue2
#                 else:
#                     local_params[varname] = reduce_func_dict[varname](varvalue1, varvalue2)
#
#             if condition_func(_self, local_params):
#                 func(_self, **local_params)
#                 for varname in _varnames_:
#                     local_params[varname] = None
#
#         return wrapper
#
#     return decorater
