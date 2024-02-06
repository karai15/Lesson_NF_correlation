import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    fc = 100e9
    wl = 3e8 / fc

    param = {
        "d": 1 / 2,
        "N": 200,
        "L": 3,
        "N_user": 1,
        "K_factor_dB": 10,
        #
        # target
        "AoA_target_start": -90 * np.pi / 180,  # theta target start
        "AoA_target_end": 90 * np.pi / 180,  # theta target end
        "r_target_start": 0.1 / wl,
        "r_target_end": 0.1 / wl,
        "AoA_spread": 15 * np.pi / 180,  # AoA spread
        "N_sector": 5,  # N_user / N_sector must be integer
        "opt_random_LoS": 1,  # 1:完全ランダム, 2:セクタ内でランダム, 0: セクタないで分解能以内にスケジューリング
    }


    N = param["N"]
    N_user = param["N_user"]

    N_rep = 1000
    Cov = np.zeros((N, N), dtype=np.complex128)
    for n_rep in range(N_rep):
        H_true, Z_set, A_set, r_set, AoA_set, cor_ant = generate_channel(param)
        Cov += H_true @ np.conj(H_true).T / N_user

    Cov_FF = np.zeros((N, N), dtype=np.complex128)
    param["r_target_start"] = 1e8 / wl
    param["r_target_end"] = 1e8 / wl
    for n_rep in range(N_rep):
        H_true, Z_set, A_set, r_set, AoA_set, cor_ant = generate_channel(param)
        Cov_FF += H_true @ np.conj(H_true).T / N_user

    D = np.fft.fft(np.eye(N)) / np.sqrt(N)  # DFT


    Cov_FF = np.conj(D).T @ Cov_FF @ D / N_rep
    Cov = np.conj(D).T @ Cov @ D / N_rep

    Cov[range(N), range(N)] = 0
    Cov_FF[range(N), range(N)] = 0

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(Cov))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(Cov_FF))
    plt.colorbar()
    plt.show()

    D = np.fft.fft(np.eye(N)) / np.sqrt(N)  # DFT
    H_true = np.conj(D).T @ H_true


    plt.plot(np.abs(H_true[:, 1]), "x-")
    plt.show()
    tet = 1




def generate_channel(param):
    # phy param
    d = param["d"]  # normalized by wavelength
    N = param["N"]
    L = param["L"]
    K_factor_dB = param["K_factor_dB"]
    N_user = param["N_user"]

    # target param
    AoA_target_start = param["AoA_target_start"]
    AoA_target_end = param["AoA_target_end"]
    r_target_start = param["r_target_start"]
    r_target_end = param["r_target_end"]
    AoA_spread = param["AoA_spread"]
    cor_ant = np.arange(N) * d - (N - 1) / 2  # coordinate of antennas

    # data
    A_set = np.zeros((N, L, N_user), dtype=np.complex128)
    H_set = np.zeros((N, N_user), dtype=np.complex128)
    AoA_set = np.zeros((L, N_user), dtype=np.float64)
    r_set = np.zeros((L, N_user), dtype=np.float64)
    Z_set = np.zeros((L, N_user), dtype=np.complex128)

    ####################################################
    # LoS
    N_div = param["N_sector"]
    opt_random_LoS = param["opt_random_LoS"]
    if opt_random_LoS == 1:  # Generate randomly
        AoA_LoS = np.random.rand(N_user) * (AoA_target_end - AoA_target_start) + AoA_target_start
    elif opt_random_LoS == 2:  # 領域をN_div個のセクタに分解して, セクタごとにランダムにLoSを生成
        AoA_LoS = generate_LoS_AoA(N_user, N_div, AoA_target_start, AoA_target_end)

    ####################################################

    # ##############
    # # TEST plot
    # # plt.plot(AoA_LoS * 180 / np.pi, "x")
    # plt.plot(AoA_LoS * 180 / np.pi, np.zeros(N_user), "x")
    # plt.show()
    # ##############

    for n_user in range(N_user):
        # NLoS
        AoA_target = np.zeros(L, dtype=np.float64)
        AoA_NLoS = np.random.rand(L - 1) * AoA_spread - AoA_spread / 2 + AoA_LoS[n_user]
        AoA_target[0] = AoA_LoS[n_user]
        AoA_target[1:] = AoA_NLoS
        AoA_set[:, n_user] = AoA_target
        #
        z_target = generate_path(L, K_factor_dB)  # (L)
        r_target = generate_distance(L, r_target_start, r_target_end)  # (L)
        A = array_factor_ula_nf(r_target, AoA_target, cor_ant)  # (N, L)  # NF
        h = A @ z_target  #

        # collect data
        A_set[:, :, n_user] = A
        H_set[:, n_user] = h
        AoA_set[:, n_user] = AoA_target
        r_set[:, n_user] = r_target
        Z_set[:, n_user] = z_target

    # ###############################
    # # TEST PLOT (θ, r)
    # wl = param["wl"]
    # # plt.plot(AoA_LoS * 180 / np.pi, r_set[0, :] * wl, "x")
    # plt.plot(np.pi * np.sin(AoA_LoS) * N / 2.783, r_set[0, :] * wl, "x")
    # plt.xlabel("AoA [degree]")
    # plt.ylabel("distance [m]")
    # plt.show()
    # ###############################

    return H_set, Z_set, A_set, r_set, AoA_set, cor_ant


def generate_path(L, K_factor_dB):
    """
    generate path based on exponential delay profile
    :param L: num of path
    :param tau_max:
    :param K_factor_dB: K_factor
    :return:
        path_delay (L, 1)
        path_power (L, 1)
    """
    K_factor = 10 ** (K_factor_dB / 10)
    path_power = np.zeros(L)

    if L >= 2:
        # NLoS
        path_power = np.abs(np.random.randn(L) + 1j * np.random.randn(L)) ** 2  # complex gaussian path
        path_power[1:L] = 1 / (K_factor + 1) * path_power[1:L] / np.sum(path_power[1:L])  # normalization
        # LoS
        path_power[0] = K_factor / (K_factor + 1)
    else:
        # LoS only
        path_power[0] = 1

    path_gain = np.sqrt(path_power) * np.exp(1j * 2 * np.pi * np.random.uniform(0, 1, size=L))  # path gain (L, 1)

    # # ##############
    # # # PLOT
    # plt.plot(path_power, "x")
    # plt.show()
    # # ##############

    return path_gain


def generate_angle(L, theta_target_start, theta_target_end):
    theta_start, theta_fin = theta_target_start, theta_target_end  # range of theta
    theta_target = (np.random.rand(L) * (theta_fin - theta_start) + theta_start)
    # theta_target = np.sort(theta_target)

    return theta_target


def generate_distance(L, rs, re):
    r_target = (np.random.rand(L) * (re - rs) + rs)  # range of target (L, 1)
    return r_target


def generate_LoS_AoA(N_user, N_div, AoA_target_start, AoA_target_end):

    N_user_div = int(N_user / N_div)
    AoA_range = AoA_target_end - AoA_target_start
    AoA_range_div = AoA_range / N_div

    AoA_LoS = np.zeros((N_div, N_user_div), dtype=np.float64)
    for n_div in range(N_div):
        AoA_s = AoA_target_start + n_div * AoA_range_div
        AoA_e = AoA_s + AoA_range_div
        AoA_LoS_div = np.random.rand(N_user_div) * (AoA_e - AoA_s) + AoA_s
        AoA_LoS[n_div, :] = AoA_LoS_div
    AoA_LoS = AoA_LoS.reshape(-1)  # (Nu)

    ##############
    # TEST plot
    # plt.plot(AoA_LoS * 180 / np.pi, "x")
    # plt.plot(AoA_LoS * 180 / np.pi, np.zeros(N_user), "x")
    # plt.show()
    ##############

    return AoA_LoS


def array_factor_ula_nf(r_v, theta_v, yn):
    """
    :param N: num of antenna
    :param r_v: radius of targets (normalized by wave length) (L, 1)
    :param theta_v: angle of targets (L, 1)
    :param yn: y-axis of antennas (L, 1)
    :return: A_nf: Array response matrix (N,L)
    """
    N = len(yn)
    if type(r_v) != np.ndarray: r_v = np.array([r_v])
    if type(theta_v) != np.ndarray: theta_v = np.array([theta_v])
    R = np.tile(r_v[None, :], (N, 1))
    Rn = np.sqrt(R ** 2 + yn[:, None] ** 2 - 2 * R * (yn[:, None] @ np.sin(theta_v[None, :])))
    # A_nf = np.exp(1j * 2 * np.pi * Rn)
    # A_nf = np.exp(1j * 2 * np.pi * (Rn - R))
    A_nf = np.exp(-1j * 2 * np.pi * (Rn - R))
    # A_nf = np.exp(-1j * 2 * np.pi * Rn)

    return A_nf  # A_nf = [a(θ1,r1), ..., a(θL, rL)]  (N, L)


main()
