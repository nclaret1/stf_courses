# conformal_cifar.py
import numpy as np
import torch
from functools import partial, reduce
from operator import itemgetter
from scipy.optimize import minimize
from itertools import starmap

# Module-level state
_state = {'alpha': 0.1}

def set_alpha(alpha):
    _state['alpha'] = alpha

def get_alpha():
    return _state['alpha']

def compose(*funcs):
    return reduce(lambda f, g: lambda x: f(g(x)), funcs[::-1])

def calculate_quantile_loss(scores, threshold, alpha=None):
    alpha = alpha or get_alpha()
    return np.add(
        np.multiply(alpha, np.maximum(np.subtract(scores, threshold), 0)),
        np.multiply(1 - alpha, np.maximum(np.subtract(threshold, scores), 0))
    )

def find_static_threshold(scores, alpha=None):
    alpha = alpha or get_alpha()
    threshold_range = np.linspace(np.min(scores) - 1, np.max(scores) + 1, 1000)
    loss_calculator = partial(calculate_quantile_loss, scores, alpha=alpha)
    losses = map(compose(np.mean, loss_calculator), threshold_range)
    return threshold_range[np.argmin(list(losses))]

def find_conditional_threshold(scores, feature_projections, alpha=None):
    alpha = alpha or get_alpha()
    X = np.column_stack([np.ones(len(feature_projections)), feature_projections])
    
    objective_func = lambda theta: np.mean(calculate_quantile_loss(scores, X @ theta, alpha))
    
    return minimize(objective_func, np.zeros(X.shape[1]), method='BFGS').x

def create_prediction_sets(logits, threshold):
    return list(map(lambda logit: np.where(logit >= threshold)[0], logits))

def load_data_to_numpy(data_loader, model):
    def process_batch(inputs, labels):
        with torch.no_grad():
            outputs = model(inputs)
        return inputs.numpy(), outputs.numpy(), labels.numpy()
    
    model.eval()
    processed_data = starmap(process_batch, data_loader)
    return map(np.vstack, zip(*processed_data))

def check_coverage(prediction_sets, true_labels):
    is_covered = lambda pred_set, true_label: true_label in pred_set
    coverage = np.mean(list(starmap(is_covered, zip(prediction_sets, true_labels))))
    avg_size = np.mean(list(map(len, prediction_sets)))
    return coverage, avg_size

# Additional utility functions
def apply_threshold(logits, threshold):
    return partial(create_prediction_sets, threshold=threshold)(logits)

def evaluate_threshold(threshold, logits, true_labels):
    prediction_sets = apply_threshold(logits, threshold)
    return check_coverage(prediction_sets, true_labels)

def optimize_threshold(logits, true_labels, alpha=None):
    set_alpha(alpha or get_alpha())
    scores = np.max(logits, axis=1)
    threshold = find_static_threshold(scores)
    return evaluate_threshold(threshold, logits, true_labels)
