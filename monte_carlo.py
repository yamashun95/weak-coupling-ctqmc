import numpy as np


# モンテカルロシミュレーションのための補助関数
def ratio_sigma(g_tau, l, sigma, x, x_l_prime, alpha):
    r = 1 + (1 - g_tau[l, l]) * (np.exp(sigma * alpha * (x[l] - x_l_prime)) - 1)
    return r


def ratio(g_up, g_dn, l, x, x_l_prime, alpha):
    return ratio_sigma(g_up, l, 1, x, x_l_prime, alpha) * ratio_sigma(
        g_dn, l, -1, x, x_l_prime, alpha
    )


def g_prime_l1l2(g, sigma, l1, l2, l, x, x_l_prime, alpha):
    return (
        g[l1, l2]
        + (g[l1, l] - (1 if l == l1 else 0))
        * (np.exp(sigma * alpha * (x[l] - x_l_prime)) - 1)
        / (1 + (1 - g[l, l]) * (np.exp(sigma * alpha * (x[l] - x_l_prime)) - 1))
        * g[l, l2]
    )


def g_prime_l1l2_vectorized(g, sigma, l, x, x_l_prime, alpha):
    exp_term = np.exp(sigma * alpha * (x[l] - x_l_prime)) - 1
    g_l_l = g[l, l]
    factor = exp_term / (1 + (1 - g_l_l) * exp_term)

    delta = np.zeros(g.shape[0], dtype=complex)
    delta[l] = 1
    delta_g = factor * np.einsum("i,j->ij", (g[:, l] - delta), g[l, :])

    return g + delta_g


def monte_carlo_sampling(g0_tau, U, delta_tau, n_tau, n_warmup=100, n_cycle=1000):
    alpha = np.arccosh(np.exp(delta_tau * U / 2))
    x = np.zeros(n_tau)
    x_prime = np.zeros(n_tau)
    g_up = g0_tau.copy()
    g_dn = g0_tau.copy()
    g_up_prime = np.zeros((n_tau, n_tau), dtype=complex)
    g_dn_prime = np.zeros((n_tau, n_tau), dtype=complex)
    g_up_average = np.zeros((n_tau, n_tau), dtype=complex)
    g_dn_average = np.zeros((n_tau, n_tau), dtype=complex)

    # 初期化
    for l in range(n_tau):
        x_l_prime = np.random.choice([-1, 1])
        x_prime[l] = x_l_prime
        # for l1, l2 in np.ndindex(n_tau, n_tau):
        #    g_up_prime[l1, l2] = g_prime_l1l2(g_up, 1, l1, l2, l, x, x_l_prime, alpha)
        #    g_dn_prime[l1, l2] = g_prime_l1l2(g_dn, -1, l1, l2, l, x, x_l_prime, alpha)
        g_up_prime = g_prime_l1l2_vectorized(g_up, 1, l, x, x_l_prime, alpha)
        g_dn_prime = g_prime_l1l2_vectorized(g_dn, -1, l, x, x_l_prime, alpha)
        x = x_prime.copy()
        g_up = g_up_prime.copy()
        g_dn = g_dn_prime.copy()

    # ウォームアップ
    for i in range(n_warmup):
        print(f"n_warmup: {i}")
        for l in range(n_tau):
            x_l_prime = -x[l]
            r = ratio(g_up, g_dn, l, x, x_l_prime, alpha)
            if np.random.rand() < r:
                x_prime[l] = x_l_prime
                # for l1, l2 in np.ndindex(n_tau, n_tau):
                #    g_up_prime[l1, l2] = g_prime_l1l2(
                #        g_up, 1, l1, l2, l, x, x_l_prime, alpha
                #    )
                #    g_dn_prime[l1, l2] = g_prime_l1l2(
                #        g_dn, -1, l1, l2, l, x, x_l_prime, alpha
                #    )
                g_up_prime = g_prime_l1l2_vectorized(g_up, 1, l, x, x_l_prime, alpha)
                g_dn_prime = g_prime_l1l2_vectorized(g_dn, -1, l, x, x_l_prime, alpha)
                x = x_prime.copy()
                g_up = g_up_prime.copy()
                g_dn = g_dn_prime.copy()

    # モンテカルロサンプリング
    for i in range(n_cycle):
        print(f"n_cycle: {i}")
        for l in range(n_tau):
            x_l_prime = -x[l]
            r = ratio(g_up, g_dn, l, x, x_l_prime, alpha)
            if np.random.rand() < r:
                x_prime[l] = x_l_prime
                # for l1, l2 in np.ndindex(n_tau, n_tau):
                #    g_up_prime[l1, l2] = g_prime_l1l2(
                #        g_up, 1, l1, l2, l, x, x_l_prime, alpha
                #    )
                #    g_dn_prime[l1, l2] = g_prime_l1l2(
                #        g_dn, -1, l1, l2, l, x, x_l_prime, alpha
                #    )
                g_up_prime = g_prime_l1l2_vectorized(g_up, 1, l, x, x_l_prime, alpha)
                g_dn_prime = g_prime_l1l2_vectorized(g_dn, -1, l, x, x_l_prime, alpha)
                x = x_prime.copy()
                g_up = g_up_prime.copy()
                g_dn = g_dn_prime.copy()
        g_up_average += g_up
        g_dn_average += g_dn

    g_up_average /= n_cycle
    g_dn_average /= n_cycle

    return g_up_average, g_dn_average
