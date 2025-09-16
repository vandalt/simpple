import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def chainplot(
    chains: np.ndarray,
    labels: list[str] | None = None,
    fig: Figure | None = None,
    axs: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot MCMC chains

    :param chains: Array of MCMC chains, with the "parameters" dimension as the last one
    :param labels: Name of the parameters
    :return: Figure and Axes for the plot
    """
    ndim = chains.shape[-1]
    if labels is None:
        labels = [f"x{i}" for i in range(ndim)]
    elif len(labels) != ndim:
        raise ValueError(
            f"Only {len(labels)} labels provided for chain of {ndim} dimensions"
        )

    if fig is None and axs is None:
        fig, axs = plt.subplots(ndim, 1, figsize=(8, ndim))
    elif fig is not None and axs is None:
        axs = fig.axes
        if isinstance(axs, Axes):
            axs = [axs]
        if len(axs) != ndim:
            raise ValueError(
                f"fig has {len(axs)} axes, but chain has {ndim} dimensions."
                "Either modify fig or use the axs argument to specify which axes to use."
            )
    elif fig is None and axs is not None:
        if isinstance(axs, Axes):
            axs = [axs]
        fig = axs[0].Figure
    if len(axs) != ndim:
        raise ValueError(
            f"The provided axes have {len(axs)} axes, but chain has {ndim} dimensions."
        )

    for i in range(ndim):
        axs[i].plot(chains[:, :, i], "k-", alpha=0.1)
        axs[i].set_ylabel(labels[i])
        if i != ndim - 1:
            axs[i].set_xticks([])
    axs[-1].set_xlabel("Steps")
    return fig, axs
