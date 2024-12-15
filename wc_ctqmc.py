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
            tau_diff = taus[i] - taus[j]
            tau_diff_mod = tau_diff % beta
            M_inv[i, j] = g(tau_diff_mod)

    # 対角要素に -delta を加える
    for i in range(n):
        M_inv[i, i] -= delta

    return M_inv


def attempt_vertex_update(C_up, C_dn, M_inv_up, M_inv_dn, g0, U, beta):
    """
    C_up, C_dnはtauのみを含むリスト。
    アップスピン挿入・除去 -> C_up操作
    ダウンスピン挿入・除去 -> C_dn操作
    """

    # 全スピン合わせた頂点数 n
    n = len(C_up) + len(C_dn)

    # コピー（受理された場合に返す）
    C_up_new = C_up.copy()
    C_dn_new = C_dn.copy()
    M_inv_up_new = M_inv_up.copy()
    M_inv_dn_new = M_inv_dn.copy()

    zeta = np.random.rand()

    if zeta < 0.5:
        # 挿入
        s = np.random.choice([+1, -1])
        tau_new = beta * np.random.rand()

        if s == +1:
            # アップスピン頂点挿入
            C_up_candidate = C_up_new + [tau_new]
            M_up_candidate = build_M_inv(C_up_candidate, g0, beta)

            # ダウン側は変更なし
            M_dn_candidate = M_inv_dn_new

            A_insert = compute_acceptance_probability_insertion(
                len(C_up_new), M_inv_up_new, M_up_candidate, U, beta
            )
            print(A_insert)

            if np.random.rand() < A_insert:
                # 受理
                C_up_new = C_up_candidate
                M_inv_up_new = M_up_candidate
                # ダウンはそのまま

        else:
            # ダウンスピン頂点挿入
            C_dn_candidate = C_dn_new + [tau_new]
            M_dn_candidate = build_M_inv(C_dn_candidate, g0, beta)

            # アップ側は変更なし
            M_up_candidate = M_inv_up_new

            A_insert = compute_acceptance_probability_insertion(
                len(C_dn_new), M_inv_dn_new, M_dn_candidate, U, beta
            )
            print(A_insert)

            if np.random.rand() < A_insert:
                C_dn_new = C_dn_candidate
                M_inv_dn_new = M_dn_candidate
                # アップはそのまま

    else:
        # 除去
        if n > 0:
            # 全頂点を合体。アップはs=+1、ダウンはs=-1とタグ付け
            C_all = [(tau, +1) for tau in C_up_new] + [(tau, -1) for tau in C_dn_new]

            idx = np.random.randint(n)
            tau_remove, s_remove = C_all[idx]

            if s_remove == +1:
                # アップスピン頂点除去
                idx_up = C_up_new.index(tau_remove)
                C_up_candidate = C_up_new.copy()
                C_up_candidate.pop(idx_up)

                M_up_candidate = build_M_inv(C_up_candidate, g0, beta)
                M_dn_candidate = M_inv_dn_new

                A_remove = compute_acceptance_probability_removal(
                    len(C_up_new), M_inv_up_new, M_up_candidate, U, beta
                )
                print(A_remove)

                if np.random.rand() < A_remove:
                    C_up_new = C_up_candidate
                    M_inv_up_new = M_up_candidate
                    # ダウンはそのまま

            else:
                # ダウンスピン頂点除去
                idx_dn = C_dn_new.index(tau_remove)
                C_dn_candidate = C_dn_new.copy()
                C_dn_candidate.pop(idx_dn)

                M_dn_candidate = build_M_inv(C_dn_candidate, g0, beta)
                M_up_candidate = M_inv_up_new

                A_remove = compute_acceptance_probability_removal(
                    len(C_dn_new), M_inv_dn_new, M_dn_candidate, U, beta
                )
                print(A_remove)

                if np.random.rand() < A_remove:
                    C_dn_new = C_dn_candidate
                    M_inv_dn_new = M_dn_candidate
                    # アップはそのまま

    return C_up_new, C_dn_new, M_inv_up_new, M_inv_dn_new


def compute_acceptance_probability_insertion(n, M_inv, M_new_inv, U, beta):
    # ダミー実装
    A_insert = -beta * U / (n + 1) * np.linalg.det(M_new_inv) / np.linalg.det(M_inv)
    return np.abs(A_insert.real)


def compute_acceptance_probability_removal(n, M_inv, M_new_inv, U, beta):
    # ダミー実装
    A_remove = -n / (beta * U) * np.linalg.det(M_new_inv) / np.linalg.det(M_inv)
    return np.abs(A_remove.real)
