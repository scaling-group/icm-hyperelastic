import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def visualize_node(mesh, nodes, node_values=None, title=None):
    """
    Visualize a scatter plot of nodes with values defined at nodes.

    Parameters
    ----------
    nodes : array-like of shape (n_nodes, 2)
        Coordinates of the mesh nodes.
    node_values : array-like of shape (n_nodes,), optional
        Values defined at each node. If None, use zero.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots()

    # If node_values is not provided, use zeros.
    if node_values is None:
        node_values = np.zeros(len(mesh))

    # Use tripcolor to fill each triangle with interpolated node values.
    vmax = np.max(np.abs(node_values))
    scatter = ax.scatter(
        nodes[:, 0], nodes[:, 1], c=node_values, cmap="coolwarm", edgecolors="none", s=30, vmin=-vmax, vmax=vmax
    )

    ax.set_aspect("equal")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.formatter = ScalarFormatter(useMathText=True)  # Use scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Set the range for scientific notation
    cbar.update_ticks()

    if title is not None:
        ax.set_title(title)

    return fig


def visualize_node_vector(mesh, nodes, node_values=None, title=None):
    """
    Visualize a vector plot of nodes with values defined at nodes.

    Parameters
    ----------
    nodes : array-like of shape (n_nodes, 2)
        Coordinates of the mesh nodes.
    node_values : array-like of shape (n_nodes,) or (n_nodes, 2), optional
        Scalar or vector values defined at each node. If None, use zero vectors.
    title : str, optional
        Title of the plot.
    normalize : bool, optional
        If True, normalize all vectors by the maximum magnitude.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots()

    # If node_values is not provided, use zero vectors.
    if node_values is None:
        node_values = np.zeros((len(nodes), 2))

    # Extract vector components
    x = nodes[:, 0]
    y = nodes[:, 1]
    u = node_values[:, 0]
    v = node_values[:, 1]

    # Calculate vector magnitudes
    magnitudes = np.sqrt(u**2 + v**2)

    # Normalize vectors
    max_magnitude = np.max(magnitudes)
    if max_magnitude > 0:
        u = u / max_magnitude
        v = v / max_magnitude

    # Plot vectors using quiver
    quiver = ax.quiver(
        x,
        y,
        u,
        v,
        magnitudes,  # x, y, u, v, color
        cmap="coolwarm",
        angles="xy",
        scale_units="xy",
        scale=10,  # arrow shorter when the value is bigger
        width=0.01,  # arrow wider when the value is bigger
        headwidth=3,  # width of the tip
        headlength=5,  # length of the tip
    )

    ax.set_aspect("equal")
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label("Magnitude")
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    if title is not None:
        ax.set_title(title)

    return fig
