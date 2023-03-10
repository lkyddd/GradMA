import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.funcs import set_seed
from lightfed.tools.model import CycleDataloader, get_parameters
from torch import nn


class ClientTrainer:
    def __init__(self, args, client_id):
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        # self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)
        self.model = None #model_pull(args).to(self.device)
        self.old_model = None
        self.global_model = None

        self.model_params = None
        self.old_model_params = None
        self.global_params = None

        self.grad_t = None
        self.old_grad_t = None
        self.global_grad_t = None

    def _zero_like(self, params):
        ans = OrderedDict()
        for name, weight in params.items():
            ans[name] = torch.zeros_like(weight, device=self.device).detach()
        return ans

    def _get_grad_(self, x, y):
        self.model.train()
        self.old_model.train()
        self.global_model.train()

        self.model.zero_grad(set_to_none=True)
        self.old_model.zero_grad(set_to_none=True)
        self.global_model.zero_grad(set_to_none=True)

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)
        old_y_pred = self.old_model(x)
        global_y_pred = self.global_model(x)

        loss = self.criterion(y_pred, y)
        old_loss = self.criterion(old_y_pred, y)
        global_loss = self.criterion(global_y_pred, y)

        loss.backward()
        old_loss.backward()
        global_loss.backward()

        grad = OrderedDict()
        old_grad = OrderedDict()
        global_grad = OrderedDict()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for name, weight in self.model.named_parameters():
                _g = weight.grad.detach()
                if 'bias' not in name:  
                    _g += (weight * self.weight_decay).detach()
                grad[name] = _g

            for name, weight in self.old_model.named_parameters():
                _g = weight.grad.detach()
                if 'bias' not in name:  
                    _g += (weight * self.weight_decay).detach()
                old_grad[name] = _g

            for name, weight in self.global_model.named_parameters():
                _g = weight.grad.detach()
                if 'bias' not in name:  
                    _g += (weight * self.weight_decay).detach()
                global_grad[name] = _g

        self.model.zero_grad(set_to_none=True)
        self.old_model.zero_grad(set_to_none=True)
        self.global_model.zero_grad(set_to_none=True)
        return grad, old_grad, global_grad

    def clear(self):
        self.model = None  # model_pull(args).to(self.device)
        self.old_model = None
        self.global_model = None

        self.model_params = None
        self.global_params = None

        self.grad_t = None
        self.old_grad_t = None
        self.global_grad_t = None

    def train_locally_step(self, step):
        self.model_params = get_parameters(self.model.state_dict(), deepcopy=True)
        logging.debug(f"train_locally_step for step: {step}")
        batch_x, batch_y = self._new_random_batch()
        self.grad_t, self.old_grad_t, self.global_grad_t = self._get_grad_(batch_x, batch_y)

    def _new_random_batch(self):
        x, y = next(self.train_batch_data_iter)
        return x, y

    def get_eval_info(self, step):
        res = {'communication round': step, 'client_id': self.client_id}
        self.model_params = get_parameters(self.model, deepcopy=True)

        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.train_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.model_params,
        #                             device=self.device)
        # res.update(train_loss=loss, train_acc=acc, train_sample_size=num)

        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.test_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.model_params,
        #                             device=self.device)
        # res.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        return res
