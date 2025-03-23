import numpy as np


def build_M_inv(C, g, beta, delta=1e-2):
    """
    スピンを考慮しないM_inv行列の構築。
    C : list of tauのみ
    g : g(tau)関数 (tau in [0,beta))
    beta : float
    delta : float (微小量)
    """
    n = len(C)
    if n == 0:
        # n=0なら M_inv = [[1.0]]
        return np.array([[1.0]], dtype=complex)

    taus = C  # tauのみのリスト
    M_inv = np.zeros((n, n), dtype=complex)

    # スピン関係なしで全ての成分を計算
    for i in range(n):
        for j in range(n):
            tau_diff = taus[i] - taus[j] - delta
            if tau_diff < 0:
                tau_diff += beta
            M_inv[i, j] = g(tau_diff)

    return M_inv


def build_S(g, beta, delta=1e-2):
    return g(beta - delta)


def build_Q(C, g, beta, tau, delta=1e-2):
    """
    スピンを考慮しないQ行列の構築。
    C : list of tauのみ
    g : g(tau)関数 (tau in [0,beta))
    beta : float
    """
    n = len(C)
    if n == 0:
        #  error
        raise ValueError("Q matrix is not defined for n=0")
    else:
        Q = np.zeros((n, 1), dtype=complex)
        for i in range(n):
            tau_diff = tau - C[i] - delta
            if tau_diff < 0:
                tau_diff += beta
            Q[i, 0] = g(tau_diff)
        return Q


def build_R(C, g, beta, tau, delta=1e-2):
    """
    スピンを考慮しないR行列の構築。
    C : list of tauのみ
    g : g(tau)関数 (tau in [0,beta))
    beta : float
    """
    n = len(C)
    if n == 0:
        #  error
        raise ValueError("R matrix is not defined for n=0")
    else:
        R = np.zeros((1, n), dtype=complex)
        for i in range(n):
            tau_diff = C[i] - tau - delta
            if tau_diff < 0:
                tau_diff += beta
            R[0, i] = g(tau_diff)
        return R


def build_S_tilde(S, R, Q, M):
    return 1 / (S - (R @ M @ Q)[0, 0])


def build_Q_tilde(S_tilde, Q, M):
    return -S_tilde * M @ Q


def build_R_tilde(S_tilde, R, M):
    return -S_tilde * R @ M


def build_P_tilde(S_tilde, R, Q, M):
    return M + S_tilde * M @ Q @ R @ M


def calculate_accept_ratio_insertion(S, Q, R, M, n, U, beta):
    A_insert = -beta * U / (n + 1) * (S - (R @ M @ Q)[0, 0])
    return np.abs(A_insert.real)


def calculate_accept_ratio_removal(S_tilde, n, U, beta):
    A_remove = -n / (beta * U) * S_tilde
    return np.abs(A_remove.real)


def build_reduced_M(S_tilde, Q_tilde, R_tilde, P_tilde):
    return P_tilde - Q_tilde @ R_tilde / S_tilde


def build_insert_M(S, Q, R, M):
    S_tilde = build_S_tilde(S, R, Q, M)
    Q_tilde = build_Q_tilde(S_tilde, Q, M)
    R_tilde = build_R_tilde(S_tilde, R, M)
    P_tilde = build_P_tilde(S_tilde, R, Q, M)
    uppper = np.hstack([P_tilde, Q_tilde])
    lower = np.hstack([R_tilde, np.array([[S_tilde]])])
    return np.vstack([uppper, lower])


def attempt_vertex_update(C, M, g0, U, beta):
    """
    C_up, C_dnはtauのみを含むリスト。
    アップスピン挿入・除去 -> C_up操作
    ダウンスピン挿入・除去 -> C_dn操作
    """

    # 全スピン合わせた頂点数 n
    n = len(C["u"]) + len(C["d"])

    zeta = np.random.rand()
    s = np.random.choice(["u", "d"])
    if zeta < 0.5:
        # 挿入
        tau_new = beta * np.random.rand()

        if len(C[s]) == 0:
            S = build_S(g0, beta)
            A_insert = np.abs((-beta * U / (n + 1) * S).real)
            if np.random.rand() < A_insert:
                # 受理
                C[s].append(tau_new)
                M[s] = np.array([[1 / S]])
                return C, M
            else:
                return C, M
        else:
            S = build_S(g0, beta)
            R = build_R(C[s], g0, beta, tau_new)
            Q = build_Q(C[s], g0, beta, tau_new)

            A_insert = calculate_accept_ratio_insertion(S, Q, R, M[s], n, U, beta)

            if np.random.rand() < A_insert:
                # 受理
                C[s].append(tau_new)
                M[s] = build_insert_M(S, Q, R, M[s])
                return C, M
            else:
                return C, M
    else:
        if len(C[s]) == 0:
            return C, M
        else:
            idx = np.random.randint(len(C[s]))
            S_tilde = M[s][idx, idx]
            A_remove = calculate_accept_ratio_removal(S_tilde, n, U, beta)

            if np.random.rand() < A_remove:
                P_tilde = np.delete(np.delete(M[s], idx, axis=0), idx, axis=1)
                Q_tilde = np.delete(M[s][:, idx : idx + 1], idx, axis=0)
                R_tilde = np.delete(M[s][idx : idx + 1, :], idx, axis=1)
                M[s] = build_reduced_M(S_tilde, Q_tilde, R_tilde, P_tilde)
                C[s].pop(idx)
                return C, M
            else:
                return C, M
