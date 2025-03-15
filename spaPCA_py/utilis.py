import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def cluster_SpatialPCs(obj, n_clusters=5, n_init=10, max_iter=300):
    SpatialPCs = obj["SpatialPCs"].T

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    cluster_labels = kmeans.fit_predict(SpatialPCs)

    obj["cluster_labels"] = cluster_labels

    return obj


def plot_cluster(
    location,
    clusterlabel,
    pointsize=3,
    text_size=15,
    title_in="",
    color_in=None,
    legend="none",
):
    location = np.array(location)
    clusterlabel = np.array(clusterlabel)

    loc_x = location[:, 0]
    loc_y = location[:, 1]

    unique_clusters = np.unique(clusterlabel)

    if color_in is None:
        cmap = plt.get_cmap("tab10")
        color_in = [cmap(i) for i in range(len(unique_clusters))]

    cluster_colors = {
        cluster: color_in[i % len(color_in)]
        for i, cluster in enumerate(unique_clusters)
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in unique_clusters:
        idx = clusterlabel == cluster
        ax.scatter(
            loc_x[idx],
            loc_y[idx],
            c=[cluster_colors[cluster]],
            label=str(cluster),
            s=pointsize,
        )

    ax.set_title(title_in, fontsize=text_size, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    if legend == "right":
        ax.legend(title="Cluster", fontsize=text_size * 0.8)

    plt.show()

    return fig
