import numpy as np
from weak_coupling_ctqmc.mesh import Meshiw, Meshitau


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


import numpy as np


def giw_to_gtau_single(giw, tau):
    """
    1つの虚時間 tau における G(tau) を計算する関数。

    パラメータ:
    -----------
    giw : Giw
        Matsubara周波数領域で定義されたグリーン関数オブジェクト。
        giw.meshiw.iw で Matsubara周波数配列 iω_n が取得可能。
        giw.meshiw.beta で βが取得可能。
        giw.giw_value で G(iω_n) の配列が取得可能。

    tau : float
        虚時間 (0 ≤ tau < β)

    戻り値:
    --------
    Gtau_value : complex
        入力した tau における G(tau) の値。
    """
    beta = giw.meshiw.beta
    iw = giw.meshiw.iw
    giw_value = giw.giw_value

    # G(tau) = (1/beta)*Σ_n G(iw_n)*exp(-i w_n tau)
    exp_factor = np.exp(-1j * iw * tau)
    Gtau_value = np.sum(giw_value * exp_factor) / beta

    return Gtau_value
