import scipy.stats as sts
from scipy import optimize
from numba import njit, jit, vectorize
import scipy.special as spec
from math import erf, sqrt
import tqdm
import random
import torch
import pandas as pd
import numpy as np

def set_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Для всех GPU
        torch.backends.cudnn.deterministic = True  # Детерминированные алгоритмы
        torch.backends.cudnn.benchmark = False  # Отключаем автотюнинг

    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)

@vectorize(['float32(float32, float32, float32)'])
@njit
def norm_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sqrt(2 * sigma))))

@njit
def SplitCoefs(coef):
    n_components = coef.shape[0] // 3
    return coef[:n_components].astype(np.float32),\
           coef[n_components: 2*n_components].astype(np.float32),\
           coef[2 * n_components :].astype(np.float32)

@njit
def CDFGmm(x, means, stds, probs):
    return np.dot(probs, norm_cdf(x, means, stds))

@njit
def quantile_function(q, coefs, x):
    means, stds, probs = SplitCoefs(coefs)
    return CDFGmm(np.float32(x), means, stds, probs) - np.float32(q)

@njit
def FindRoute(q, coefs):
    l = -1000
    r = 1000
    while r - l > 1e-4:
        mid = np.float32((l + r) / 2)
        if (quantile_function(q, coefs, mid) < 0):
            l = mid
        else:
            r = mid
    return l

@njit
def GetQuantiles(coef):
    quantiles = np.zeros(9, dtype=np.float32)
    for ind, quantile in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        quantiles[ind] = FindRoute(quantile, coef)
    return quantiles

@njit
def GetCDFs(coef, n_points):
    cdfs = np.zeros(n_points, dtype=np.float32)
    for ind, point in enumerate(np.linspace(-5, 5, n_points).astype(np.float32)):
        means, stds, probs = SplitCoefs(coef)
        cdfs[ind] = CDFGmm(point, means, stds, probs)
    return cdfs

@njit
def GetAB(row):
    a, b, p = SplitCoefs(row)
    return np.array([
        (a * p).sum(),
        (b * p).sum()
    ], dtype=np.float32)

# @njit
def ProcessCoefs(coefs):
    new_coefs = np.zeros((coefs.shape[0], 9), dtype=np.float32)
    for i in tqdm.tqdm(range(new_coefs.shape[0])):
        new_coefs[i, :] = GetQuantiles(coefs[i, :])
        # break
    return new_coefs


def ProcessCoefsCDF(coefs, n_points=7):
    new_coefs = np.zeros((coefs.shape[0], n_points), dtype=np.float32)
    for i in tqdm.tqdm(range(new_coefs.shape[0])):
        new_coefs[i, :] = GetCDFs(coefs[i, :], n_points)
    return new_coefs

def ProcessCoefsAB(coefs):
    coefs = coefs.astype(np.float32)
    new_coefs = np.zeros((coefs.shape[0], 2), dtype=np.float32)
    for i in tqdm.tqdm(range(new_coefs.shape[0])):
        new_coefs[i, :] = GetAB(coefs[i, :])
    return new_coefs
