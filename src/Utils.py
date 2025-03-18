import copy
import torch

from torch.utils.data import Dataset


class ICUData(Dataset):
    def __init__(self, dataframe, vitals_cols, labs_cols, label_col):
        self.vitals = dataframe[vitals_cols].values
        self.labs = dataframe[labs_cols].values
        self.labels = dataframe[label_col].values

        self.vitals = torch.tensor(self.vitals, dtype=torch.float32)
        self.labs = torch.tensor(self.labs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vitals[idx], self.labs[idx], self.labels[idx]


def compute_distance(state_dict1, state_dict2, p=2):
    """
    Tính khoảng cách giữa hai state_dict của PyTorch bằng norm Lp.

    Args:
        state_dict1 (dict): state_dict đầu tiên.
        state_dict2 (dict): state_dict thứ hai.
        p (int): Bậc của norm (mặc định là L2 norm).

    Returns:
        float: Khoảng cách giữa hai state_dict.
    """
    total_distance = 0.0

    for key in state_dict1.keys():
        if key in state_dict2:
            diff = state_dict1[key] - state_dict2[key]
            diff = diff.reshape(-1)  # Chuyển tensor về dạng 1D
            total_distance += torch.linalg.norm(diff, ord=p).item()

    return total_distance


def create_random_base_model(state_dict, perturbation=0.1):
    base_model = copy.deepcopy(state_dict)
    for param in base_model.keys():
        if base_model[param].dtype != torch.long:
            base_model[param] += torch.randn_like(base_model[param]) * perturbation

    return base_model


def check_min_max_distance(all_genuine_models, max_distance, malicious_model):
    malicious_distance = 0.0

    for genuine_model in all_genuine_models:
        distance = compute_distance(malicious_model, genuine_model)
        if distance > malicious_distance:
            malicious_distance = distance

    return malicious_distance < max_distance


def check_min_sum_distance(all_genuine_models, max_distance, malicious_model):
    malicious_distance = 0.0

    for genuine_model in all_genuine_models:
        malicious_distance += compute_distance(malicious_model, genuine_model) ** 2

    return malicious_distance < max_distance


### ATTACK SCHEMES


def calculate_mean_and_std(state_dicts):
    mean_std_dict = {}

    for key in state_dicts[0].keys():
        if state_dicts[0][key].dtype != torch.long:
            params = torch.stack([state_dict[key] for state_dict in state_dicts])
            mean_val = torch.mean(params, dim=0)
            std_val = torch.std(params, dim=0)
            sign_val = torch.sign(mean_val)
            mean_std_dict[key] = {
                'mean': mean_val,
                'std': std_val,
                'sign': sign_val
            }

    return mean_std_dict


def create_opt_fang_model(state_dict, all_genuine_models, gamma=50.0, tau=1.0):
    if len(all_genuine_models) <= 1:
        return state_dict

    malicious_model = None
    mean_std_dict = calculate_mean_and_std(all_genuine_models)

    max_distance = 0.0
    for i in range(len(all_genuine_models) - 1):
        for j in range(i + 1, len(all_genuine_models)):
            distance = compute_distance(all_genuine_models[i], all_genuine_models[j])
            if distance > max_distance:
                max_distance = distance

    step = gamma
    gamma_succ = 0.0

    while abs(gamma_succ - gamma) > tau:
        print(f"Gamma is {gamma}")
        malicious_model = all_genuine_models[0]
        for key, stats in mean_std_dict.items():
            # benign_mean + gamma * perturbation
            malicious_model[key] = stats['mean'] - gamma * stats['sign']

        if check_min_max_distance(all_genuine_models, max_distance, malicious_model):
            gamma_succ = gamma
            gamma = gamma + step / 2
        else:
            gamma = gamma - step / 2
        step = step / 2

    return malicious_model


def create_min_max_model(state_dict, all_genuine_models, gamma=50.0, tau=1.0):
    if len(all_genuine_models) <= 1:
        return state_dict

    malicious_model = None
    mean_std_dict = calculate_mean_and_std(all_genuine_models)

    max_distance = 0.0
    for i in range(len(all_genuine_models) - 1):
        for j in range(i + 1, len(all_genuine_models)):
            distance = compute_distance(all_genuine_models[i], all_genuine_models[j])
            if distance > max_distance:
                max_distance = distance

    step = gamma
    gamma_succ = 0.0

    while abs(gamma_succ - gamma) > tau:
        print(f"Gamma is {gamma}")
        malicious_model = all_genuine_models[0]
        for key, stats in mean_std_dict.items():
            # benign_mean + gamma * perturbation
            malicious_model[key] = stats['mean'] - gamma * stats['std']

        if check_min_max_distance(all_genuine_models, max_distance, malicious_model):
            gamma_succ = gamma
            gamma = gamma + step / 2
        else:
            gamma = gamma - step / 2
        step = step / 2

    return malicious_model


def create_min_sum_model(state_dict, all_genuine_models, gamma=50.0, tau=1.0):
    if len(all_genuine_models) <= 1:
        return state_dict

    malicious_model = None
    mean_std_dict = calculate_mean_and_std(all_genuine_models)

    max_distance = 0.0
    for i in range(len(all_genuine_models)):
        sum_distance = 0.0
        for j in range(len(all_genuine_models)):
            if i == j:
                continue
            sum_distance += compute_distance(all_genuine_models[i], all_genuine_models[j]) ** 2

        if sum_distance > max_distance:
            max_distance = sum_distance

    step = gamma
    gamma_succ = 0.0

    while abs(gamma_succ - gamma) > tau:
        print(f"Gamma is {gamma}")
        malicious_model = all_genuine_models[0]
        for key, stats in mean_std_dict.items():
            # benign_mean + gamma * perturbation
            malicious_model[key] = stats['mean'] - gamma * stats['std']

        if check_min_sum_distance(all_genuine_models, max_distance, malicious_model):
            gamma_succ = gamma
            gamma = gamma + step / 2
        else:
            gamma = gamma - step / 2
        step = step / 2

    return malicious_model


def create_LIE_state_dict(list_dict, scaling_factor=0.74):
    mean_std_dict = calculate_mean_and_std(list_dict)
    malicious_state_dict = list_dict[0]

    for key, stats in mean_std_dict.items():
        malicious_state_dict[key] = stats['mean'] + scaling_factor * stats['std']

    return malicious_state_dict
