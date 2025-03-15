import math
import numpy as np
from scipy.sparse.linalg import eigsh


def SpatialPCA_SpatialPCs(params, fast=False, eigenvecnum=None):

    # construct U and delta
    if not fast:
        U = params["U"]
        delta = params["delta"]
    else:
        if eigenvecnum is not None:
            print("Low rank approximation!")
            print(
                f"Using user selected top {eigenvecnum} eigenvectors and eigenvalues in the Kernel matrix!"
            )
            delta, U = eigsh(params["kernelmat"], k=eigenvecnum, which="LM")
        elif params["n"] > 5000:
            fast_eigen_num = math.ceil(params["n"] * 0.1)
            print("Low rank approximation!")
            print(
                "Large sample, using top 10% sample size of eigenvectors and eigenvalues in the Kernel matrix!"
            )
            delta, U = eigsh(params["kernelmat"], k=fast_eigen_num, which="LM")
        else:
            U = params["U"]
            delta = params["delta"]
            ind = len(delta)
            print("Low rank approximation!")
            print(
                f"Small sample, using top {ind} eigenvectors and eigenvalues in the Kernel matrix!"
            )

    params["U"] = U
    params["delta"] = delta

    W_hat_t = params["W"].T
    WtYM = W_hat_t @ params["YM"]
    WtYMK = WtYM @ params["kernelmat"]
    WtYMU = WtYM @ U

    Ut = U.T
    UtM = Ut @ params["M"]
    UtMK = UtM @ params["kernelmat"]
    UtMU = UtM @ U

    middle_matrix = (1.0 / params["tau"]) * np.diag(1.0 / delta) + UtMU
    middle_inv = np.linalg.inv(middle_matrix)

    SpatialPCs = params["tau"] * WtYMK - params["tau"] * (WtYMU @ middle_inv @ UtMK)
    params["SpatialPCs"] = SpatialPCs

    return params
