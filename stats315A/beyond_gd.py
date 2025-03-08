# beyond_gd_starter.py
#
# Starter code for methods beyond stochastic gradient / gradient-type
# methods. You should change the name of this file to beyond_gd.py
# when you use it in your code, so that mnist_experiments.py fits it
# correctly.

import torch.optim as optim
from torch.optim import Adagrad, SGD
from typing import cast, List, Optional, Union
import torch
from torch import Tensor
from torch.optim import Optimizer

from torch.optim.adagrad import adagrad as adagrad

class TruncatedSGD(Optimizer):
    """Implements truncated stochastic gradient method"""
    
    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        """Perform single optimization step with truncated SGD"""
        if closure is None:
            raise ValueError("Requires closure computing loss, not None")
        else:
            with torch.enable_grad():
                loss = closure()

        apprentissage_base = self.param_groups[0]["lr"]
        apprentissage_corrige = apprentissage_base
        norme_gradient_racine = 0.0


        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norme_gradient_racine += torch.norm(p.grad.data)**2
                    ratio =loss.item()/norme_gradient_racine
                    apprentissage_corrige = min(apprentissage_corrige, ratio)
                   
                    
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if (p.grad is not None):
                    p.data.add_(p.grad.data, alpha = -apprentissage_corrige)
        return loss



class TruncatedAdagrad(Optimizer):
    """Implements a truncated AdaGrad method extending Truncated SGD"""

    def __init__(self, params, lr=1e-2, eps=1e-10):
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)

        # Initialize sum of squared gradients
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {"squared_sum": torch.zeros_like(p)}

    def step(self, closure=None):
        """Perform a single optimization step for TruncatedAdagrad"""
        if closure is None:
            raise ValueError("Requires closure computing loss, not None")
        
        with torch.enable_grad():
            loss = closure()

        apprentissage_base = self.param_groups[0]["lr"]
        eps = self.param_groups[0]["eps"]

        apprentissage_corrige = apprentissage_base
        norme_gradient_transforme = 0.0


        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    state["squared_sum"].add_(p.grad.data**2)
                    deviation = state["squared_sum"].sqrt() + eps
                    norme_gradient_transforme += torch.sum(((p.grad.data)**2)/deviation)
                    ratio = loss.item() / norme_gradient_transforme
                    apprentissage_corrige = min(apprentissage_corrige, ratio)
                    
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if (p.grad is not None):
                    std = state["squared_sum"].sqrt() + eps
                    p.data.addcdiv_(p.grad, std, value = -apprentissage_corrige)
        return loss
