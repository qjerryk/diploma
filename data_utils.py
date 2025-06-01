import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
tqdm.pandas()

def compute_eigenvalues_symmetric(matrix):
    """
    Вычисляет собственные значения симметричной матрицы.
    
    Параметры:
    ----------
    matrix : numpy.ndarray
        Симметричная квадратная матрица.
    
    Возвращает:
    -----------
    eigenvalues : numpy.ndarray
        Упорядоченные собственные значения (по убыванию).
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной!")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Матрица не симметричная!")
    # print(matrix)
    eigenvalues = np.linalg.eigvalsh(matrix)
    # print(matrix, eigenvalues)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    
    return eigenvalues_sorted

def compute_wasserstein(row):
    means = row[['a_1', 'a_2', 'a_3', 'a_4']].values
    stds = row[['b_1', 'b_2', 'b_3', 'b_4']].values
    probs = row[[f'probs_{i}' for i in range(1, 5)]].values
    n_components = len(means)
    wasserstein_matrix = np.zeros((n_components, n_components))
    
    for i in range(n_components):
        for j in range(n_components):
            wasserstein_matrix[i, j] = np.clip(np.sqrt((means[i] - means[j]) ** 2 + (stds[i] - stds[j])**2) / (1 - probs[i] * probs[j]), 0, 1e3)
    
    return wasserstein_matrix

def get_matr_feats(row):
    matr = compute_wasserstein(row)
    res = compute_eigenvalues_symmetric((matr + matr.T) / 2)
    return list(res)


class DataTs(Dataset):
    def __init__(self, df, seq_length, column, forecast_horizon=500, use_matrix_features=False):
        self.seq_length = seq_length
        if use_matrix_features:
            eig_values = df.progress_apply(get_matr_feats, axis=1, result_type='expand')
            eig_values.columns = [f'eig_{i}' for i in range(1, 5)]
            df = pd.concat([df, eig_values], axis=1)
            df = df.drop(columns=[f'probs_{i}' for i in range(1, 5)] + [f'a_{i}' for i in range(1, 5)] + [f'b_{i}' for i in range(1, 5)])
        self.data = df
        # print('post_proc_matrix columns', df.columns)
        self.X = torch.tensor(self.data.values, dtype=torch.float32)
        self.y = torch.tensor(self.data[column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, index):
        return self.X[index : index + self.seq_length], self.y[index + self.seq_length]

