import numpy as np
import random
import copy

import torch


def non_iid_rate(num_data, rate):
    result = []
    for _ in range(num_data):
        if rate < random.random():
            result.append(0)
        else:
            result.append(1)
    return np.array(result)


def create_random_base_model(model, perturbation=0.1):
    base_model = copy.deepcopy(model)
    for param in base_model.keys():
        if base_model[param].dtype != torch.long:
            base_model[param] += torch.randn_like(base_model[param]) * perturbation

    return base_model
