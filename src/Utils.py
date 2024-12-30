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


def compute_distance(state_dict1, state_dict2):
    """
    Compute distance between two state_dict.

    Args:
        state_dict1 (dict): first state_dict.
        state_dict2 (dict): second state_dict.

    Returns:
        float: Total distance L2 between two weights.
    """
    total_distance = 0.0

    for key in state_dict1.keys():
        param1 = state_dict1[key].float()
        param2 = state_dict2[key].float()

        distance = torch.norm(param1 - param2)
        total_distance += distance.item()

    return total_distance


### ATTACK SCHEMES


def create_random_base_model(state_dict, perturbation=0.1):
    base_model = copy.deepcopy(state_dict)
    for param in base_model.keys():
        if base_model[param].dtype != torch.long:
            base_model[param] += torch.randn_like(base_model[param]) * perturbation

    return base_model


def create_min_max_model(state_dict, all_genuine_models, step=0.001):
    max_distance = 0.0

    if len(all_genuine_models) <= 1:
        return state_dict

    for i in range(len(all_genuine_models) - 1):
        for j in range(i + 1, len(all_genuine_models)):
            distance = compute_distance(all_genuine_models[i], all_genuine_models[j])
            if distance > max_distance:
                max_distance = distance

    distance_step = 0.0
    malicious_distance = 0.0
    malicious_model = None
    while malicious_distance < max_distance:
        distance_step += step
        malicious_model = create_random_base_model(state_dict, distance_step)
        # Calculate distance to all genuine models
        for genuine_model in all_genuine_models:
            distance = compute_distance(malicious_model, genuine_model)
            if distance > malicious_distance:
                malicious_distance = distance

    print(distance_step)
    return malicious_model


def create_min_sum_model(state_dict, all_genuine_models, step=0.001):
    max_distance = 0.0

    for i in range(len(all_genuine_models)):
        sum_distance = 0.0
        for j in range(len(all_genuine_models)):
            if i == j:
                continue
            sum_distance += compute_distance(all_genuine_models[i], all_genuine_models[j])

        if sum_distance > max_distance:
            max_distance = sum_distance

    distance_step = 0.0
    malicious_distance = 0.0
    malicious_model = None
    while malicious_distance < max_distance:
        distance_step += step
        malicious_model = create_random_base_model(state_dict, distance_step)

        sum_distance = 0.0
        # Calculate distance to all genuine models
        for genuine_model in all_genuine_models:
            sum_distance += compute_distance(malicious_model, genuine_model)

        if sum_distance > malicious_distance:
            malicious_distance = sum_distance

    return malicious_model

