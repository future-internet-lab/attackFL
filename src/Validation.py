import torch
import numpy as np
import gzip
import pickle
import torch.nn.functional as F
import math
import torchvision
import torchvision.transforms as transforms

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

        if hasattr(src.Model, model_name):
            self.model = getattr(src.Model, model_name)()
        else:
            raise ValueError(f"Model name '{model_name}' is not valid.")

        if self.data_name:
            # Load test_dataset
            if self.data_name == "ICU":
                with gzip.open("test_dataset.pkl.gz", "rb") as f:
                    test_dataset = pickle.load(f)
            elif self.data_name == "CIFAR10":
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
                test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                            transform=transform_test)
            else:
                raise ValueError(f"Data name '{data_name}' is not valid.")

            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    def test(self, final_state_dict, device):
        try:
            self.model.load_state_dict(final_state_dict)
        except Exception as e:
            src.Log.print_with_color(f"[Warning] Cannot load state dict for testing. {e}", "yellow")
            for name, param in self.model.named_parameters():
                param.data = final_state_dict[name]
        self.model.to(device)
        # evaluation mode
        self.model.eval()
        if self.data_name == "ICU":
            return self.test_icu(device)
        elif self.data_name == "CIFAR10":
            return self.test_image(device)
        else:
            raise ValueError(f"Not found test function for data name {self.data_name}")

    def test_image(self, device):
        test_loss = 0
        correct = 0
        for data, target in tqdm(self.test_loader):
            data = data.to(device)
            target = target.to(device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), accuracy))
        self.logger.log_info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), accuracy))

        if np.isnan(test_loss) or math.isnan(test_loss) or abs(test_loss) > 10e5:
            return False

        return True

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
                if torch.isnan(outputs).any():
                    src.Log.print_with_color("NaN detected in output, training false", "yellow")
                    return False

                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Concatenate all labels and outputs
        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        # Compute ROC-AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
        roc_auc = auc(fpr, tpr)

        print(f"ROC_AUC: {roc_auc:.4f}")
        self.logger.log_info(f"ROC_AUC: {roc_auc:.4f}")

        return True
