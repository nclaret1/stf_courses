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
from scipy.sparse import issparse
import numpy as np

from torch.optim.adagrad import adagrad as adagrad

class TruncatedSGD(Optimizer):
    """Implements truncated stochastic gradient method"""
    
    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform single optimization step with truncated SGD"""
        if closure is None:
            raise ValueError("Requires closure computing loss, not None")
        
        # Compute the loss
        with torch.enable_grad():
            loss = closure()

        base_lr = self.param_groups[0]["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g_k = p.grad.data  
                    step = -base_lr * g_k

                    if loss.item() + torch.dot(g_k.view(-1), step.view(-1)) < 0:
                        step = torch.zeros_like(step)  

                    p.data.add_(step)

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

        eps = self.param_groups[0]["eps"]

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    #get sum of squared gradients for param p
                    state = self.state[p]
                    #get gradient of paramter p
                    g_k = p.grad.data 
                    #add currrent covariance 
                    state["squared_sum"].add_(g_k**2)
                    #take sqrt element-wiseto obtain H_k vector
                    H_k = state["squared_sum"].sqrt() + eps
                    step = - (group["lr"] / H_k) * g_k
                    if loss.item() + torch.dot(g_k.view(-1), step.view(-1)) < 0:
                        step = torch.zeros_like(step) 

                    p.data.add_(step)

        return loss

class LossFunction:
    def __init__(self):
        """Initializes the loss function"""

    def zero_coefficient(self, X, y):
        """Returns an appropriate vector or matrix of all-zeros for the problem
        """
        return np.zeros(X.shape[1])

    def minimal_value(self):
        """Returns minimal value of an instantaneous loss

        This is useful when implementing proximal-type methods, which (e.g.)
        wish to iterate via

        theta_new = argmin_theta {
           max{ l(theta_init) + g' * (theta - theta_init), 0} +
           1 / (2 * stepsize) * norm(theta - theta_init)^2 },

        a type of proximal-like update. Here, 0 would normally be replaced
        by the minimal value the loss can take, and g is a (sub)gradient
        of the loss at theta_init.
        """
        return 0.0
    
    def loss(self, x_i, y_i, theta):
        """Computes the given loss on the input data pair
        
        Parameters
        ----------
          x_i: A single datapoint (covariates)
          y_i: The label / target
          theta: The vector or matrix (as appropriate) of parameters
        """
        return 0.0;

    def sample_loss(self, X, y, theta):
        """Computes the average loss on the entire sample

        Parameters
        ----------
          X: The n-by-dim matrix of all data
          y: The vector (or whatever) of labels
        """
        return 0.0;
    
    def gradient(self, x_i, y_i, theta):
        """Computes the gradient of the loss of interest
        
        """
        return 0.0 * x_i;

    def prox_update(self, x_i, y_i, theta_init, alpha):
        """Computes a proximal-point update for the loss.

        Parameters
        ----------
          x_i: The datapoint (vector of covariates) x_i 
          y_i: The label y_i
          theta_init: The initial parameter vector theta
          alpha: Scalar in the below minimization problem

        Computes the (stochastic) proximal point update

        theta' = argmin_theta { l(theta, x_i, y_i) + ...
                       (1 / (2 * alpha)) * norm(theta - theta_init)^2 }

        and returns the updated parameter vector theta' solving it.

        In the base loss function class, this raises a
        NotImplementedError to enforce implementation if using this
        method.

        """
        raise NotImplementedError

class BinaryLogisticLoss(LossFunction):
    """Implements the binary logistic loss, assuming labels are +/- 1 valued.

    """
    
    def loss(self, x_i, y_i, theta):
        """Computes logistic loss

        Computes logistic loss on example pair x_i, y_i at the
        parameter theta

        """
        arg = y_i * (x_i @ theta)

        if arg > 0:
            return np.log(1 + np.exp(-arg))
        else:
            return -arg + np.log(1 + np.exp(arg))
        
    def gradient(self, x_i, y_i, theta):
        """Computes gradient of binary logistic loss
        """
        arg = None
        
        if issparse(x_i):
            arg = y_i * (x_i.data @ theta[x_i.indices])
        else:
            arg = y_i * (x_i @ theta)
        
        # compute sigmoid
        if arg >= 0:
            z = np.exp(-arg)
            sigmoid = 1 / (1 + z)
        else:
            z = np.exp(arg)
            sigmoid = z / (1 + z)
        return - y_i * (1 - sigmoid) * x_i
        
    def sample_loss(self, X, y, theta):
        """Computes average logistic loss on dataset

        Computes the average of the logistic losses across the dataset
        in the pair (X, y)
        
        """
        arg = y * (X @ theta)

        pos_mask = arg > 0
        neg_mask = ~pos_mask

        loss = np.zeros_like(arg)
        loss[pos_mask] = np.log(1 + np.exp(-arg[pos_mask]))
        loss[neg_mask] = -arg[neg_mask] + np.log(1 + np.exp(arg[neg_mask]))

        return np.mean(loss)
    
