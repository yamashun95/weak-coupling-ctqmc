import numpy as np
from scipy.integrate import quad


# ベーテ格子の密度分布関数 D(ε)
def bethe_density_of_states(epsilon, W, D_0):
    return (D_0 / W) * np.sqrt(W**2 - epsilon**2)


# グリーン関数 G(ω) の実部と虚部
def bethe_green_function_real(omega, W, D_0, delta):
    integrand = lambda epsilon: np.real(
        bethe_density_of_states(epsilon, W, D_0) / (omega - epsilon + 1j * delta)
    )
    return np.real(quad(integrand, -W, W)[0])


def bethe_green_function_imag(omega, W, D_0, delta):
    integrand = lambda epsilon: np.imag(
        bethe_density_of_states(epsilon, W, D_0) / (omega - epsilon + 1j * delta)
    )
    return quad(integrand, -W, W)[0]


def bethe_green_function(omega, W, D_0, delta):
    G_real = bethe_green_function_real(omega, W, D_0, delta)
    G_imag = bethe_green_function_imag(omega, W, D_0, delta)
    return G_real + 1j * G_imag
