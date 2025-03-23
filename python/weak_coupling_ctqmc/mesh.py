import numpy as np


# Matsubara周波数格子
class Meshiw:
    def __init__(self, beta, niw_max):
        self.beta = beta
        n_values = np.arange(-niw_max, niw_max)  # nの範囲を決定
        self.iw = (2 * n_values + 1) * np.pi / beta


# imaginary time格子
class Meshitau:
    def __init__(self, beta, n_tau):
        self.beta = beta
        self.n_tau = n_tau
        self.tau = (np.arange(n_tau) + 1 / 2) * beta / n_tau
