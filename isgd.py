from torch.optim.optimizer import Optimizer, required
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import defaultdict
import math
import pdb

class ISGD(torch.optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0: 
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                       nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening!")
        
        super(ISGD, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(ISGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    @staticmethod
    def compute_pseudo_gradient(parameters, lr):
        theta = torch.cat([x.grad.data.flatten() for x in parameters]).cpu()
        H = torch.cat([(i * theta).unsqueeze(dim=-1) for i in theta], dim=1)
        U = torch.eye(H.size(0)) + lr * H
        pseudo_grad, _ = torch.gesv(theta, U)
        return pseudo_grad.cuda()
    
    @staticmethod
    def reconstruct_gradients(parameters, pseudo_grad):
        gradients = defaultdict()
        ctr = 0
        for param in parameters:
            length = 1
            for i in range(param.dim()):
                length *= param.grad.size(i)
            gradients[param] = pseudo_grad[ctr:ctr+length].reshape(param.grad.size())
            ctr+=length
        return gradients
    
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            lr = group['lr']
            if lr == 0:
                continue
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            params = [x for x in group['params'] if x.grad is not None]
            pseudo_grad = self.compute_pseudo_gradient(params, lr)
            gradients = self.reconstruct_gradients(params, pseudo_grad)
            
            for p in params:
                d_p = gradients[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1-dampening, d_p)
                    if nesterov:
                        d_p = d_p.add_(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-lr, d_p)
                
        return loss

class IAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(IAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(IAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    @staticmethod
    def compute_pseudo_gradient(parameters, lr):
        theta = torch.cat([x.grad.data.flatten() for x in parameters]).cpu()
        H = torch.cat([(i * theta).unsqueeze(dim=-1) for i in theta], dim=1)
        U = torch.eye(H.size(0)) + lr * H
        try:
            pseudo_grad, _ = torch.gesv(theta, U)
        except:
            pseudo_grad = torch.zeros(theta.size())
        return pseudo_grad.cuda()
    
    @staticmethod
    def reconstruct_gradients(parameters, pseudo_grad):
        gradients = defaultdict()
        ctr = 0
        for param in parameters:
            length = 1
            for i in range(param.dim()):
                length *= param.grad.size(i)
            gradients[param] = pseudo_grad[ctr:ctr+length].reshape(param.grad.size())
            ctr+=length
        return gradients
    
    def step(self, closure=None): 
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            if lr == 0:
                continue
            params = [x for x in group['params'] if x.grad is not None]
            pseudo_grad = self.compute_pseudo_gradient(params, lr)
            gradients = self.reconstruct_gradients(params, pseudo_grad)
            
            for p in params:
                grad = gradients[p]
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss    
    
    