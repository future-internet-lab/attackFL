import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import cosine_similarity


def normalize_weights(w):
    norm = np.linalg.norm(w)
    return w / norm if norm != 0 else w


def ShieldFL(all_model_parameters):
    weights = [a['weight'] for a in all_model_parameters]
    weights_np = np.array([w.numpy() for w in weights])
    weights_np = np.array([normalize_weights(w) for w in weights_np])

    # Tính toán cosine similarity giữa từng weight và trung bình toàn bộ tập hợp
    mean_weight = np.mean(weights_np, axis=0)
    cos_similarities = cosine_similarity(weights_np, mean_weight.reshape(1, -1)).flatten()

    # Xác định ngưỡng phát hiện độc hại
    cosine_threshold = np.percentile(cos_similarities, 20)  # 20% thấp nhất có thể là tấn công
    malicious_detected_cosine = cos_similarities < cosine_threshold

    return np.where(malicious_detected_cosine)[0]
