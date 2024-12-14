import numpy as np
from mesh import Meshiw, Meshitau


# Matsubara周波数グリーン関数
class Giw:
    def __init__(self, meshiw, giw_value):
        self.meshiw = meshiw
        self.giw_value = giw_value


# imaginary timeグリーン関数
class Gtau:
    def __init__(self, meshitau, gtau_value):
        self.meshitau = meshitau
        self.gtau_value = gtau_value


# Matsubara周波数からimaginary timeへの変換
def make_Gtau_from_Giw(giw, n_tau):
    beta = giw.meshiw.beta
    iw = giw.meshiw.iw
    meshitau = Meshitau(beta, n_tau)
    tau_values = meshitau.tau
    Gtau_value = [
        np.sum(giw.giw_value * np.exp(-1j * iw * tau)) / beta for tau in tau_values
    ]
    return Gtau(meshitau, Gtau_value)
