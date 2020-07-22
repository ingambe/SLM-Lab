from abc import ABC, abstractmethod
from slm_lab.agent.net import net_util
from collections import deque
import pydash as ps
import torch
import torch.nn as nn


class Net(ABC):
    '''Abstract Net class to define the API methods'''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        @param {dict} net_spec is the spec for the net
        @param {int|list} in_dim is the input dimension(s) for the network. Usually use in_dim=body.state_dim
        @param {int|list} out_dim is the output dimension(s) for the network. Usually use out_dim=body.action_dim
        '''
        self.net_spec = net_spec
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_norms = None  # for debugging
        self.previous_grads = None
        if not ps.is_empty(self.net_spec.get('avg_grad_clipper')):
            avg_grad_clipper = self.net_spec.get('avg_grad_clipper')
            self.grad_scaler = float(avg_grad_clipper['avg_scaler'])
            stack_size = int(avg_grad_clipper['n_last'])
            self.previous_grads = deque(maxlen=stack_size)
        if self.net_spec.get('gpu'):
            if torch.cuda.device_count():
                self.device = f'cuda:{net_spec.get("cuda_id", 0)}'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

    @abstractmethod
    def forward(self):
        '''The forward step for a specific network architecture'''
        raise NotImplementedError

    @net_util.dev_check_train_step
    def train_step(self, loss, optim, lr_scheduler=None, clock=None, global_net=None):
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=ps.get(clock, 'frame'))
        optim.zero_grad()
        loss.backward()
        if self.previous_grads is not None:
            total_norm = 0.0
            for p in self.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.previous_grads.append(total_norm)
            average_grad = sum(self.previous_grads) / len(self.previous_grads)
            if self.clip_grad_val is not None:
                nn.utils.clip_grad_norm_(self.parameters(), min(average_grad * self.grad_scaler, self.clip_grad_val))
            else:
                nn.utils.clip_grad_norm_(self.parameters(), average_grad * self.grad_scaler)
        elif self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        if global_net is not None:
            net_util.push_global_grads(self, global_net)
        optim.step()
        if global_net is not None:
            net_util.copy(global_net, self)
        if clock is not None:
            clock.tick('opt_step')
        return loss

    def store_grad_norms(self):
        '''Stores the gradient norms for debugging.'''
        norms = [param.grad.norm().item() for param in self.parameters()]
        self.grad_norms = norms
