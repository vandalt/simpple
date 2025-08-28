import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def chainplot(
    chains: np.ndarray, labels: list[str] | None = None
) -> tuple[Figure, Axes]:
    """Plot MCMC chains

    :param chains: Array of MCMC chains, with the "parameters" dimension as the last one
    :param labels: Name of the parameters
    :return: Figure and Axes for the plot
    """
    ndim = chains.shape[-1]
    if labels is None:
        labels = [f"x{i}" for i in range(ndim)]
    fig, axs = plt.subplots(ndim, 1)
    for i in range(ndim):
        axs[i].plot(chains[:, :, i], "k-", alpha=0.1)
        axs[i].set_ylabel(labels[i])
    axs[-1].set_xlabel("Steps")
    return fig, axs
