import logging
import math
import os

import numpy as np
import pandas as pd
import torch
from collections import OrderedDict

from experiments.models.model import model_pull
from lightfed.core import BaseServer
from lightfed.tools.aggregator import ModelStateAvgAgg
from lightfed.tools.funcs import (consistent_hash, formula, save_pkl, set_seed)
from lightfed.tools.model import evaluation, get_buffers, get_parameters, get_cpu_param, QP_project_op, QP_project_ne
from torch import nn

from trainer import ClientTrainer

import cvxopt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


class ServerManager(BaseServer):

    def __init__(self, ct, args):
        super().__init__(ct)
        self.super_params = args.__dict__.copy()
        self.app_name = args.app_name
        self.device = args.device
        self.client_num = args.client_num
        self.selected_client_num = args.selected_client_num
        self.I = args.I
        self.eval_step_interval = args.eval_step_interval
        self.eval_on_full_test_data = args.eval_on_full_test_data

        self.gamma_l_list = args.gamma_l_list.copy()
        self.gamma_l_list.append((math.inf, None))  
        self.gamma_l = 0.0
        self.next_gamma_l_stage = -1

        self.gamma_g_list = args.gamma_g_list.copy()
        self.gamma_g_list.append((math.inf, None))  
        self.gamma_g = 0.0
        self.next_gamma_g_stage = -1

        self.full_train_dataloader = args.data_distributer.get_train_dataloader()  
        self.full_test_dataloader = args.data_distributer.get_test_dataloader()    
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)
        self.model = model_pull(args).to(self.device)  
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if not os.path.exists(f"{path}/model_save/{args.model_type}.pth"):
            torch.save(self.model, f"{path}/model_save/{args.model_type}.pth")
        _initial_model_buffer = get_buffers(self.model.state_dict(), deepcopy=True) 
        self._model_with_batchnorm = bool(len(_initial_model_buffer))
        logging.info(f"model_with_batchnorm: {self._model_with_batchnorm}")

        self.global_params_buffer = get_cpu_param(self.model.state_dict())
        self.global_params = get_parameters(self.global_params_buffer, deepcopy=True)
        torch.cuda.empty_cache()

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]

        self.global_grad = None
        self.global_grad_aggregator = ModelStateAvgAgg()
        self.global_buffer_aggregator = ModelStateAvgAgg()

        self.client_eval_info = []  
        self.global_eval_info = []  

        self.comm_round = args.comm_round
        self.unfinished_client_num = -1

        self.step = -1

    def _zero_like(self, params):
        ans = OrderedDict()
        for name, weight in params.items():
            if 'weight' in name or 'bias' in name:
                ans[name] = torch.zeros_like(weight, device=self.device).detach()
        return ans

    def start(self):
        logging.info("start...")
        self.next_step()

    def end(self):
        logging.info("end...")


        del self.super_params['data_distributer']
        del self.super_params['log_level']
        self.super_params['device'] = self.super_params['device'].type

        ff = f"{self.app_name}-{consistent_hash(self.super_params, code_len=64)}.pkl"
        logging.info(f"output to {ff}")

        result = {'super_params': self.super_params,
                  'global_eval_info': pd.DataFrame(self.global_eval_info),
                  'client_eval_info': pd.DataFrame(self.client_eval_info)}
        save_pkl(result, f"{os.path.dirname(__file__)}/Result/{ff}")

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1
        self._set_gamma_l(self.step)
        self._set_gamma_g(self.step)
        self.selected_clients = self._new_train_workload_arrage()  
        self.unfinished_client_num = self.selected_client_num
        self.global_grad_aggregator.clear()
        self.global_buffer_aggregator.clear()

        for client_id in self.selected_clients:
            self._ct_.get_node('client', client_id) \
                .fed_client_train_step(step=self.step, gamma_l=self.gamma_l, global_params_buffer=self.global_params_buffer)

    def _new_train_workload_arrage(self):
        selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        return selected_client

    def _set_gamma_l(self, step):
        if step >= self.next_gamma_l_stage:
            _, self.gamma_l = self.gamma_l_list.pop(0)
            self.next_gamma_l_stage = self.gamma_l_list[0][0]
            logging.info(f"update gamma_l to :{self.gamma_l}, next_gamma_l_stage:{self.next_gamma_l_stage}")

    def _set_gamma_g(self, step):
        if step >= self.next_gamma_g_stage:
            _, self.gamma_g = self.gamma_g_list.pop(0)
            self.next_gamma_g_stage = self.gamma_g_list[0][0]
            logging.info(f"update gamma_g to :{self.gamma_g}, next_gamma_g_stage:{self.next_gamma_g_stage}")

    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_model_params_buffer):
        # logging.debug(f"train comm. round of client_id:{client_id} comm. round:{step} was finished")
        assert self.step == step

        weight = self.local_sample_numbers[client_id]

        if self._model_with_batchnorm:
            model_buffer = get_buffers(client_model_params_buffer, deepcopy=True)
            self.global_buffer_aggregator.put(model_buffer, weight)

        client_model_params = get_parameters(client_model_params_buffer, deepcopy=True)
        client_acc_g = formula(lambda gp, cmp: gp - cmp,
                               self.global_params, client_model_params)
        self.global_grad_aggregator.put(client_acc_g, weight)

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.global_grad = self.global_grad_aggregator.get_and_clear()

            self.global_params = formula(lambda p_t, g_t: p_t - self.gamma_g * g_t,
                                         self.global_params, self.global_grad)

            if self._model_with_batchnorm:
                self.global_buffer = self.global_buffer_aggregator.get_and_clear()
                for name, param in self.global_params_buffer.items():
                    if 'weight' in name or 'bias' in name:
                        self.global_params_buffer[name] = self.global_params[name].detach()
                    else:
                        self.global_params_buffer[name] = self.global_buffer[name].detach()
                self.model.load_state_dict(self.global_params_buffer)
            else:
                self.global_params_buffer = get_parameters(self.global_params, deepcopy=True)
                self.model.load_state_dict(self.global_params_buffer)

            if self.I == 1:
                if self.step % self.eval_step_interval == 0:
                    self._set_global_eval_info()
            else:
                self._set_global_eval_info()

            logging.debug(f"train comm. round:{step} is finished")
            self.next_step()

    def _set_global_eval_info(self):

        eval_info = {'comm. round': self.step}
        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.full_train_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.global_params,
        #                             device=self.device,
        #                             eval_full_data=False)
        # eval_info.update(train_loss=loss, train_acc=acc, train_sample_size=num)

        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_test_dataloader,
                                    criterion=self.criterion,
                                    device=self.device,
                                    eval_full_data=self.eval_on_full_test_data)
        torch.cuda.empty_cache()
        eval_info.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        logging.info(f"global eval info:{eval_info}")
        self.global_eval_info.append(eval_info)

class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.rho_l = args.rho_l
        self.I = args.I
        self.device = args.device
        self.client_id = self._ct_.role_index
        self.model_type = args.model_type

        self.trainer = ClientTrainer(args, self.client_id)

        self.step = 0
        self.gamma_l = 0.0


    def start(self):
        logging.info("start...")

    def end(self):
        logging.info("end...")

    def end_condition(self):
        return False

    def fed_client_train_step(self, step, gamma_l, global_params_buffer):
        self.step = step
        self.gamma_l = gamma_l
        logging.debug(f"training client_id:{self.client_id}, comm. round:{step}")

        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        self.trainer.model = torch.load(f"{path}/model_save/{self.model_type}.pth")
        self.trainer.model.load_state_dict(global_params_buffer, strict=True)
        self.trainer.global_params = get_parameters(self.trainer.model.state_dict(), deepcopy=True)

        self.trainer.old_model = torch.load(f"{path}/model_save/{self.model_type}.pth")
        if self.trainer.old_model_params == None:
            self.trainer.old_model.load_state_dict(global_params_buffer, strict=True)
        else:
            self.trainer.old_model.load_state_dict(self.trainer.old_model_params, strict=True)

        self.trainer.global_model = torch.load(f"{path}/model_save/{self.model_type}.pth")
        self.trainer.global_model.load_state_dict(global_params_buffer, strict=True)

        for tau in range(self.I):
            self.trainer.train_locally_step(tau)

            local_param_diff = formula(lambda m_p, g_p: m_p - g_p,
                               self.trainer.model_params, self.trainer.global_params)

            tilde_g = self._zero_like(self.trainer.grad_t)
            for name in self.trainer.grad_t:
                params = self.trainer.grad_t[name]
                params_size = params.size()
                params_numpy = params.reshape(-1).cpu().numpy()

                G_init = np.zeros((3, len(params_numpy)))
                G_init[0] = self.trainer.old_grad_t[name].reshape(-1).cpu().numpy()
                G_init[1] = self.trainer.global_grad_t[name].reshape(-1).cpu().numpy()
                G_init[2] = local_param_diff[name].reshape(-1).cpu().numpy()

                try:
                    g_ = QP_project_op(G_init, params_numpy, params_size, rho=self.rho_l)
                    tilde_g[name] = g_.cuda(0)
                except:
                    tilde_g[name] = params

            model_params_mid = formula(lambda p_t, g_t: p_t - self.gamma_l * g_t,
                                       self.trainer.model_params, tilde_g)
            self.trainer.model.load_state_dict(model_params_mid, strict=False)
            self.trainer.old_model.load_state_dict(self.trainer.model_params, strict=False)

        self.trainer.old_model_params = self.trainer.model.state_dict()

        model_params_buffer_cpu = get_cpu_param(self.trainer.model.state_dict())
        self.finish_train_step(model_params_buffer_cpu)
        self.trainer.clear()
        torch.cuda.empty_cache()

    def _zero_like(self, params):
        ans = OrderedDict()
        for name, weight in params.items():
            if 'weight' in name or 'bias' in name:
                ans[name] = torch.zeros_like(weight, device=self.device).detach()
        return ans

    def finish_train_step(self, model_params_buffer):
        # eval_info = self.trainer.get_eval_info(self.step)
        logging.debug(f"finish_train_step_1 comm. round:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          model_params_buffer)

