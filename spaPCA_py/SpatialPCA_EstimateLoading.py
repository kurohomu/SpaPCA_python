import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler


def spatialPCA_estimate_loading(
    kernelmat,
    location,
    expr,
    covariate=None,
    maxiter=300,
    # initial_tau=1,
    fast=False,
    eigenvecnum=None,
    SpatialPCnum=20,
):
    np.random.seed(42)
    # param_ini = np.log(initial_tau)
    params = {}
    params["SpatialPCnum"] = SpatialPCnum

    scaler = StandardScaler()
    X = scaler.fit_transform(location)
    params = {}
    params["X"] = X
    params["n"] = X.shape[0]
    params["p"] = X.shape[1]
    params["expr"] = expr
    params["kernelmat"] = kernelmat

    # construct H (X in paper)
    if covariate is None:
        H = np.ones((params["n"], 1))
        params["q"] = 1
    else:
        covariate = np.array(covariate)
        if covariate.ndim == 1:
            covariate = covariate.reshape(-1, 1)
        q = covariate.shape[1] + 1
        H = np.zeros((params["n"], q))
        H[:, 0] = 1
        H[:, 1:] = covariate
        params["q"] = q
    params["H"] = H

    # construct M = I - H (H^T H)^{-1} H^T
    HH_inv = np.linalg.inv(H.T @ H)
    HH = H @ HH_inv @ H.T
    params["M"] = np.eye(params["n"]) - HH

    # construct tr_YMY and YM
    params["tr_YMY"] = np.trace(expr @ params["M"] @ expr.T)
    params["YM"] = expr @ params["M"]

    # decomposite kernelmat
    if not fast:
        print("Eigen decomposition on kernel matrix!")
        delta_all, U_all = eigh(kernelmat)
        idx = np.argsort(delta_all)[::-1]
        delta_all = delta_all[idx]
        U_all = U_all[:, idx]
        params["delta"] = delta_all
        params["U"] = U_all
        print("Using all eigenvectors and eigenvalues in the Kernel matrix!")
    else:
        print("Eigen decomposition on kernel matrix!")
        if eigenvecnum is not None:
            params["eigenvecnum"] = eigenvecnum
            if issparse(kernelmat):
                eigvals, eigvecs = eigsh(kernelmat, k=eigenvecnum)
            else:
                eigvals, eigvecs = eigsh(kernelmat, k=eigenvecnum, which="LM")
            params["delta"] = eigvals
            params["U"] = eigvecs
            print("Low rank approximation!")
            print(
                f"Using user selected top {eigenvecnum} eigenvectors and eigenvalues in the Kernel matrix!"
            )
        elif params["n"] > 5000:
            if issparse(kernelmat):
                eigvals, eigvecs = eigsh(kernelmat, k=20)
            else:
                eigvals, eigvecs = eigsh(kernelmat, k=20, which="LM")
            params["delta"] = eigvals
            params["U"] = eigvecs
            print("Low rank approximation!")
            print(
                "Large sample, using top 20 eigenvectors and eigenvalues in the Kernel matrix!"
            )
        else:
            delta_all, U_all = eigh(kernelmat)
            idx = np.argsort(delta_all)[::-1]
            delta_all = delta_all[idx]
            U_all = U_all[:, idx]
            cumsum_vals = np.cumsum(delta_all / len(delta_all))
            ind = np.searchsorted(cumsum_vals, 0.9) + 1
            print("Low rank approximation!")
            print(
                f"Small sample, using top {ind} eigenvectors and eigenvalues in the Kernel matrix!"
            )
            params["delta"] = delta_all[:ind]
            params["U"] = U_all[:, :ind]

    params["MYt"] = params["M"] @ expr.T
    params["YMMYt"] = params["YM"] @ params["MYt"]
    params["YMU"] = params["YM"] @ params["U"]
    params["Xt"] = H.T
    params["XtU"] = params["Xt"] @ params["U"]
    params["Ut"] = params["U"].T
    params["UtX"] = params["Ut"] @ H
    params["YMX"] = params["YM"] @ H
    params["UtU"] = params["Ut"] @ params["U"]
    params["XtX"] = params["Xt"] @ H
    params["SpatialPCnum"] = SpatialPCnum

    # optimize tau
    res = minimize_scalar(
        spatialPCA_estimate_parameter,
        args=(params,),
        bounds=(-10, 10),
        method="bounded",
        options={"maxiter": maxiter},
    )
    tau = np.exp(res.x)
    params["tau"] = tau

    k = expr.shape[0]
    n = expr.shape[1]
    q = params["q"]

    # calculate W
    tauD_UtU_inv = np.linalg.inv(tau * np.diag(params["delta"]) + params["UtU"])
    YMU_tauD_UtU_inv_Ut = params["YMU"] @ tauD_UtU_inv @ params["Ut"]
    YMU_tauD_UtU_inv_UtX = YMU_tauD_UtU_inv_Ut @ H
    XtU_inv_UtX = params["XtU"] @ tauD_UtU_inv @ params["UtX"]
    left = params["YMX"] - YMU_tauD_UtU_inv_UtX
    right = left.T
    middle = np.linalg.inv(-XtU_inv_UtX)
    G_each = (
        params["YMMYt"]
        - (params["YMU"] @ tauD_UtU_inv @ params["MYt"])
        - left @ middle @ right
    )

    W = eigsh(G_each, k=SpatialPCnum, which="LM")[1]
    params["W"] = W

    sigma2_0 = (params["tr_YMY"] + F_funct_sameG(W, G_each)) / (k * (n - q))
    params["sigma2_0"] = sigma2_0

    params["params"] = params

    return params


def spatialPCA_estimate_parameter(param, params):
    tau = np.exp(param)
    k = params["expr"].shape[0]
    n = params["expr"].shape[1]
    q = params["q"]
    PCnum = params["SpatialPCnum"]

    A = tau * np.diag(params["delta"]) + params["UtU"]
    tauD_UtU_inv = np.linalg.inv(A)
    YMU_tauD_UtU_inv_Ut = params["YMU"] @ tauD_UtU_inv @ params["Ut"]
    YMU_tauD_UtU_inv_UtX = YMU_tauD_UtU_inv_Ut @ params["H"]
    XtU_inv_UtX = params["XtU"] @ tauD_UtU_inv @ params["UtX"]
    left = params["YMX"] - YMU_tauD_UtU_inv_UtX
    right = left.T
    middle = np.linalg.inv(-XtU_inv_UtX)
    G_each = (
        params["YMMYt"]
        - (params["YMU"] @ tauD_UtU_inv @ params["MYt"])
        - left @ middle @ right
    )

    A1 = (1 / tau) * np.diag(1 / params["delta"]) + params["UtU"]
    sign1, logdet1 = np.linalg.slogdet(A1)
    B = tau * np.diag(params["delta"])
    sign2, logdet2 = np.linalg.slogdet(B)
    log_det_tauK_I = logdet1 + logdet2

    C = params["UtU"] + (1 / tau) * np.diag(1 / params["delta"])
    inv_C = np.linalg.inv(C)
    Xt_invmiddle_X = params["XtX"] - params["XtU"] @ inv_C @ params["UtX"]
    sign3, logdet3 = np.linalg.slogdet(Xt_invmiddle_X)
    log_det_Xt_inv_X = logdet3

    sum_det = (0.5 * log_det_tauK_I + 0.5 * log_det_Xt_inv_X) * PCnum

    W_est_here = eigsh(G_each, k=PCnum, which="LM")[1]

    obj_val = sum_det + (k * (n - q) / 2) * np.log(
        params["tr_YMY"] + F_funct_sameG(W_est_here, G_each)
    )
    return obj_val


def F_funct_sameG(X, G):
    total = 0.0
    for i in range(X.shape[1]):
        xi = X[:, i]
        total += xi.T @ G @ xi
    return -total
