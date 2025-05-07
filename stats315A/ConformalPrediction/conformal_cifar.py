# conformal_cifar.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from cifar_processing import load_numpy_into_data, split_into_train_and_validation
from conformal_multiclass_starter import train_prediction_model, evaluate
from scipy.optimize import minimize
from functools import wraps
from contextlib import contextmanager

class QuantileLossCalculator:
    @staticmethod
    def calculate_quantile_loss(scores, threshold, alpha=0.1):
        return alpha * np.maximum(scores - threshold, 0) + (1 - alpha) * np.maximum(threshold - scores, 0)

class ThresholdFinder:
    @staticmethod
    def find_static_threshold(scores, alpha=0.1):
        threshold_range = np.linspace(np.min(scores) - 1, np.max(scores) + 1, 1000)
        return threshold_range[np.argmin([np.mean(QuantileLossCalculator.calculate_quantile_loss(scores, t, alpha)) for t in threshold_range])]

    @staticmethod
    def find_conditional_threshold(scores, feature_projections, alpha=0.1):
        X = np.column_stack([np.ones(len(feature_projections)), feature_projections])
        
        objective_func = lambda theta: np.mean(QuantileLossCalculator.calculate_quantile_loss(scores, X @ theta, alpha))
        
        return minimize(objective_func, np.zeros(X.shape[1]), method='BFGS').x

class PredictionSetCreator:
    @staticmethod
    def create_prediction_sets(logits, threshold):
        return [np.where(logit >= threshold)[0] for logit in logits]

class DataLoader:
    @staticmethod
    @contextmanager
    def model_eval_context(model):
        model.eval()
        with torch.no_grad():
            yield

    @staticmethod
    def load_data_to_numpy(data_loader, model):
        features, logits, labels = [], [], []
        
        with DataLoader.model_eval_context(model):
            for inputs, batch_labels in data_loader:
                outputs = model(inputs)
                features.append(inputs.numpy())
                logits.append(outputs.numpy())
                labels.append(batch_labels.numpy())
        
        return map(lambda x: np.vstack(x) if len(x[0].shape) > 1 else np.concatenate(x),
                   (features, logits, labels))

class CoverageChecker:
    @staticmethod
    def check_coverage(prediction_sets, true_labels):
        coverage = np.mean([true_label in pred_set for pred_set, true_label in zip(prediction_sets, true_labels)])
        avg_size = np.mean([len(pred_set) for pred_set in prediction_sets])
        return coverage, avg_size

def calculate_quantile_loss(*args, **kwargs):
    return QuantileLossCalculator.calculate_quantile_loss(*args, **kwargs)

def find_static_threshold(*args, **kwargs):
    return ThresholdFinder.find_static_threshold(*args, **kwargs)

def find_conditional_threshold(*args, **kwargs):
    return ThresholdFinder.find_conditional_threshold(*args, **kwargs)

def create_prediction_sets(*args, **kwargs):
    return PredictionSetCreator.create_prediction_sets(*args, **kwargs)

def load_data_to_numpy(*args, **kwargs):
    return DataLoader.load_data_to_numpy(*args, **kwargs)

def check_coverage(*args, **kwargs):
    return CoverageChecker.check_coverage(*args, **kwargs)
