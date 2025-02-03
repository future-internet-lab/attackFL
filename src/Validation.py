import torch
import numpy as np
import gzip
import pickle

from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

import src.Model
import src.Utils
import src.Log

from tqdm import tqdm


class Validation:
    def __init__(self, model_name, data_name, logger):
        self.data_name = data_name
        self.logger = logger
        self.model = None

        if model_name == "RNNModel":
            self.model = src.Model.RNNModel(7, 16)
        else:
            raise ValueError(f"Model name '{model_name}' is not valid.")

        self.test_loader = None
        if self.data_name and not self.test_loader:
            # Load test_dataset
            if self.data_name == "ICU":
                with gzip.open("test_dataset.pkl.gz", "rb") as f:
                    test_dataset = pickle.load(f)
            else:
                raise ValueError(f"Data name '{data_name}' is not valid.")

        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    def test(self, avg_state_dict, device):
        self.model.load_state_dict(avg_state_dict)
        # evaluation mode
        self.model.eval()
        if self.data_name == "ICU":
            return self.test_icu(device)
        else:
            raise ValueError(f"Not found test function for data name {self.data_name}")

    def test_icu(self, device):
        # List to store true labels and predicted scores
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for vitals, labs, labels in tqdm(self.test_loader):
                vitals = vitals.to(device)
                labs = labs.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = self.model(vitals, labs)

                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Concatenate all labels and outputs
        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
        roc_auc = auc(fpr, tpr)

        print(f"False Positive Rate: {fpr:.4f}, True Positive Rate: {tpr:.4f}, ROC_AUC: {roc_auc:.4f}")
        self.logger.log_info(f"False Positive Rate: {fpr:.4f}, True Positive Rate: {tpr:.4f}, ROC_AUC: {roc_auc:.4f}")
        return True
