from sklearn.datasets import make_circles
import torch

n_samples = 1000

X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
