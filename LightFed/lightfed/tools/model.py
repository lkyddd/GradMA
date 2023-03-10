from collections import OrderedDict
import torch
import copy
import cvxopt
from cvxopt import matrix, solvers
import numpy as np


def evaluation(model, dataloader, criterion, model_params=None, device=None, eval_full_data=True):
    if model_params is not None:
        model.load_state_dict(model_params)

    # if device is not None:
    #     model.to(device)

    model.eval()
    loss = 0.0
    acc = 0.0
    num = 0

    for x, y in dataloader:
        torch.cuda.empty_cache()
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        with torch.no_grad():
            raw_output = model(x)
            _loss = criterion(raw_output, y)
            _, predicted = torch.max(raw_output, -1)
            _acc = predicted.eq(y).sum()
            _num = y.size(0)
            loss += (_loss * _num).item()
            acc += _acc.item()
            num += _num
            if not eval_full_data:
                break
    loss /= num
    acc /= num
    return loss, acc, num


def get_parameters(params_model, deepcopy=True):
    ans = OrderedDict()
    for name, params in params_model.items():
        if deepcopy:
            if 'weight' in name or 'bias' in name:
                params = params.clone().detach()
                ans[name] = params
    return ans


def get_buffers(params_model, deepcopy=True):
    ans = OrderedDict()
    for name, buffers in params_model.items():
        if deepcopy:
            if 'weight' in name or 'bias' in name:
                continue
            buffers = buffers.clone().detach()
            ans[name] = buffers
    return ans

def get_cpu_param(param):
    ans = OrderedDict()
    for name, param_buffer in param.items():
        ans[name] = param_buffer.clone().detach().cpu()
    torch.cuda.empty_cache()
    return ans

def get_gpu_param(param, device=None):
    ans = OrderedDict()
    for name, param_buffer in param.items():
        ans[name] = param_buffer.clone().detach().to(device)
    return ans

# def QP_project(G_not_0, not_0_row, params_numpy, params_size):
#     Q = matrix(np.dot(G_not_0, G_not_0.transpose()), tc='d')
#     p = matrix(np.dot(np.expand_dims(params_numpy, axis=0), G_not_0.transpose())[0], tc='d')
#     G = matrix(-1 * np.eye(len(not_0_row)), tc='d')
#     h = matrix(np.zeros(len(not_0_row)), tc='d')
#     sol = solvers.qp(Q, p, G, h)
#
#     v = np.dot(G_not_0.transpose(), np.array(sol['x'])).transpose()[0] + params_numpy
#     params_ = torch.Tensor(v).view(*params_size)#.to(self.device)
#     return params_

def QP_project_op(G_not_0, params_numpy, params_size, rho=0):
    len_ = len(G_not_0)
    cal_M = np.dot(G_not_0, G_not_0.transpose())
    diag_M = np.diagonal(cal_M)**0.5
    norm_2_d = np.linalg.norm(params_numpy, ord=2)
    p_1 = np.dot(np.expand_dims(params_numpy, axis=0), G_not_0.transpose())[0]
    p_2 = rho * norm_2_d * diag_M

    Q = matrix(cal_M, tc='d')
    p = matrix(p_1-p_2, tc='d')
    G = matrix(-1 * np.eye(len_), tc='d')
    h = matrix(np.zeros(len_), tc='d')
    sol = solvers.qp(Q, p, G, h)

    v = params_numpy + np.dot(G_not_0.transpose(), np.array(sol['x'])).transpose()[0]
    params_ = torch.Tensor(v).view(*params_size)#.to(self.device)
    return params_

def QP_project_ne(G_not_0, params_numpy, params_size, rho=-1*10**(-9)):
    len_ = len(G_not_0)
    cal_M = np.dot(G_not_0, G_not_0.transpose())
    diag_M = np.diagonal(cal_M)**0.5
    norm_2_d = np.linalg.norm(params_numpy, ord=2)
    p_1 = np.dot(np.expand_dims(params_numpy, axis=0), G_not_0.transpose())[0]
    p_2 = rho * norm_2_d * diag_M
    Q = matrix(cal_M, tc='d')
    p = matrix(-1 * (p_1-p_2), tc='d')
    G = matrix(-1 * np.eye(len_), tc='d')
    h = matrix(np.zeros(len_), tc='d')
    sol = solvers.qp(Q, p, G, h)

    v = params_numpy - np.dot(G_not_0.transpose(), np.array(sol['x'])).transpose()[0]
    params_ = torch.Tensor(v).view(*params_size)#.to(self.device)
    return params_



class CycleDataloader:
    def __init__(self, dataloader, epoch=-1, seed=None) -> None:
        self.dataloader = dataloader
        self.epoch = epoch
        self.seed = seed
        self._data_iter = None
        self._init_data_iter()

    def _init_data_iter(self):
        if self.epoch == 0:
            raise StopIteration()

        if self.seed is not None:
            torch.manual_seed(self.seed + self.epoch)
        self._data_iter = iter(self.dataloader)
        self.epoch -= 1

    def __next__(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._init_data_iter()
            return next(self._data_iter)

    def __iter__(self):
        return self
