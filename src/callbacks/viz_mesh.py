import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.ticker import ScalarFormatter


def visualize_mesh(mesh, nodes, triangle_values=None, linewidth=None, title=None):
    """
    Visualize a 2D triangular mesh.

    Parameters
    ----------
    mesh : array-like of shape (n_elements, 3)
        Each row contains indices into `nodes` for one triangle.
    nodes : array-like of shape (n_nodes, 2)
        Coordinates of the mesh nodes.
    triangle_values : array-like of shape (n_elements,), optional
        Values to be used for coloring the triangles. If None, use zero.
    linewidth : float, optional
        Line width for the edges of the triangles.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots()

    # Create a triangulation from nodes and connectivity.
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], mesh)

    # Create an array of values (one per triangle) to drive the facecolors.

    if triangle_values is None:
        triangle_values = np.zeros(len(mesh))

    # Use tripcolor to fill each triangle with a color.
    # The 'edgecolors' argument draws a black outline.
    vmax = np.max(np.abs(np.array(triangle_values)))
    tpc = ax.tripcolor(
        triangulation, facecolors=triangle_values, edgecolors="k", cmap="coolwarm", vmin=-vmax, vmax=vmax
    )

    # adjust the line width or color if desired.
    if linewidth is not None:
        ax.triplot(triangulation, color="k", lw=linewidth)

    ax.set_aspect("equal")
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.formatter = ScalarFormatter(useMathText=True)  # Use scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Set the range for scientific notation
    cbar.update_ticks()

    if title is not None:
        ax.set_title(title)

    return fig
