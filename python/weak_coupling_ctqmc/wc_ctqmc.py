import numpy as np


def alpha(s, sigma, delta=1e-2):
    return 0.5 + sigma * s * (0.5 + delta)


def build_M_inv(C, sigma, g, beta, eta=1e-2, delta=1e-2):
    n = len(C)
    if n == 0:
        return np.array([[1.0]], dtype=complex)

    M_inv = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            tau_diff = C[i][0] - C[j][0] - eta
            if tau_diff < 0:
                tau_diff += beta
            M_inv[i, j] = g(tau_diff)
        M_inv[i, i] -= alpha(C[i][1], sigma, delta)
    return M_inv


def build_S(g, s, sigma, beta, eta=1e-2, delta=1e-2):
    return g(beta - eta) - alpha(s, sigma, delta)


def build_Q(C, g, beta, tau):
    if len(C) == 0:
        raise ValueError("Q is undefined for empty configuration")
    Q = np.array([[g((c[0] - tau + beta) % beta)] for c in C], dtype=complex)
    return Q


def build_R(C, g, beta, tau):
    if len(C) == 0:
        raise ValueError("R is undefined for empty configuration")
    R = np.array([[g((tau - c[0] + beta) % beta) for c in C]], dtype=complex)
    return R


def build_S_tilde(S, Q, R, M):
    return 1 / (S - (R @ M @ Q)[0, 0])


def build_Q_tilde(S_tilde, Q, M):
    return -S_tilde * M @ Q


def build_R_tilde(S_tilde, R, M):
    return -S_tilde * R @ M


def build_P_tilde(S_tilde, R, Q, M):
    return M + S_tilde * M @ Q @ R @ M


def build_insert_M(S, Q, R, M):
    S_tilde = build_S_tilde(S, Q, R, M)
    Q_tilde = build_Q_tilde(S_tilde, Q, M)
    R_tilde = build_R_tilde(S_tilde, R, M)
    P_tilde = build_P_tilde(S_tilde, R, Q, M)
    return np.block([[P_tilde, Q_tilde], [R_tilde, np.array([[S_tilde]])]])


def build_reduced_M(S_tilde, Q_tilde, R_tilde, P_tilde):
    return P_tilde - (Q_tilde @ R_tilde) / S_tilde


def calculate_accept_ratio_insertion(S_up, S_dn, Q, R, M, n, U, beta):
    term_up = S_up - (R @ M["u"] @ Q)[0, 0]
    term_dn = S_dn - (R @ M["d"] @ Q)[0, 0]
    A_insert = -beta * U / (n + 1) * term_up * term_dn
    print(
        f"[DEBUG] A_insert (raw): {A_insert}  Re: {A_insert.real}, Im: {A_insert.imag}"
    )
    return np.abs(A_insert.real)


def calculate_accept_ratio_removal(S_tilde_up, S_tilde_dn, n, U, beta):
    A_remove = -n / (beta * U) * S_tilde_up * S_tilde_dn

    print(
        f"[DEBUG] A_remove (raw): {A_remove}  Re: {A_remove.real}, Im: {A_remove.imag}"
    )
    return np.abs(A_remove.real)


def attempt_vertex_update(C, M, g0, U, beta, delta=1e-2):
    n = len(C)
    move_type = np.random.choice(["insert", "remove"])
    s = np.random.choice([1, -1])
    tau_new = beta * np.random.rand()

    if move_type == "insert":
        S_up = build_S(g0, s, 1, beta, delta)
        S_dn = build_S(g0, s, -1, beta, delta)

        if n == 0:
            A_insert = np.abs((-beta * U * S_up * S_dn).real)
            if np.random.rand() < A_insert:
                C.append((tau_new, s))
                M = {"u": np.array([[1 / S_up]]), "d": np.array([[1 / S_dn]])}
        else:
            Q = build_Q(C, g0, beta, tau_new)
            R = build_R(C, g0, beta, tau_new)
            A_insert = calculate_accept_ratio_insertion(S_up, S_dn, Q, R, M, n, U, beta)
            if np.random.rand() < A_insert:
                C.append((tau_new, s))
                M["u"] = build_insert_M(S_up, Q, R, M["u"])
                M["d"] = build_insert_M(S_dn, Q, R, M["d"])
        return C, M

    elif move_type == "remove" and n > 0:
        idx = np.random.randint(n)
        S_tilde_up = M["u"][idx, idx]
        S_tilde_dn = M["d"][idx, idx]
        A_remove = calculate_accept_ratio_removal(S_tilde_up, S_tilde_dn, n, U, beta)

        if np.random.rand() < A_remove:
            C.pop(idx)
            for spin in ["u", "d"]:
                P = np.delete(np.delete(M[spin], idx, axis=0), idx, axis=1)
                Q = np.delete(M[spin][:, idx].reshape(-1, 1), idx, axis=0)
                R = np.delete(M[spin][idx, :].reshape(1, -1), idx, axis=1)
                M[spin] = build_reduced_M(M[spin][idx, idx], Q, R, P)
        return C, M

    return C, M
