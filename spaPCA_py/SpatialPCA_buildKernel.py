import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def build_kernel(location, bandwidth_set_by_user, kerneltype="gaussian"):

    scaler = StandardScaler()
    location = scaler.fit_transform(location)

    # build kernal matrix
    dist_matrix = cdist(location, location, metric="euclidean")
    if kerneltype == "gaussian":
        return np.exp(-(dist_matrix**2) / bandwidth_set_by_user)
    elif kerneltype == "cauchy":
        return 1 / (1 + (dist_matrix**2) / bandwidth_set_by_user)
    elif kerneltype == "quadratic":
        return 1 - (dist_matrix**2) / (dist_matrix**2 + bandwidth_set_by_user)
    else:
        raise ValueError("Unsupported kernel type.")
