"""Matplotlib plotting helpers for aperture objects.

Plotting is deliberately kept out of the Rust-backed computation path.
Matplotlib is imported lazily only when these helpers are called.

The helpers support the object methods implemented by ``PixelAp`` (`plot()`
and `to_patches()`) while keeping optional plotting code separate from fast
aperture-sum imports.
"""

from __future__ import annotations

from collections.abc import Iterable


def to_patch_list(patches):
    """Return a patch or patch sequence as a plain list.

    Parameters
    ----------
    patches : matplotlib patch or iterable of matplotlib patches
        Result returned by an aperture object's ``to_patches()`` method.

    Returns
    -------
    list
        A list of Matplotlib patch objects.
    """
    if isinstance(patches, Iterable) and not hasattr(patches, "set_figure"):
        return list(patches)
    return [patches]


def plot_apertures(apertures, ax=None, origin=(0, 0), **kwargs):
    """Plot one aperture object and return its Matplotlib patch object(s).

    Parameters
    ----------
    apertures : PixelAp
        Any astroapers aperture or annulus object.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If omitted, ``matplotlib.pyplot.gca()`` is used.
    origin : tuple[float, float], optional
        Pixel-coordinate origin to subtract from aperture positions. Use this
        when plotting a cutout extracted from a larger image.
    **kwargs
        Forwarded to the Matplotlib patch constructors.

    Returns
    -------
    matplotlib.patches.Patch or list[matplotlib.patches.Patch]
        A single patch for scalar-position apertures, otherwise a list.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    patches = apertures.to_patches(origin=origin, **kwargs)
    for patch in to_patch_list(patches):
        ax.add_patch(patch)
    return patches


def plot_many(apertures, ax=None, origin=(0, 0), **kwargs):
    """Plot several aperture objects on one axes.

    Parameters
    ----------
    apertures : iterable
        Iterable of astroapers aperture or annulus objects.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If omitted, ``matplotlib.pyplot.gca()`` is used.
    origin : tuple[float, float], optional
        Pixel-coordinate origin to subtract from all aperture positions.
    **kwargs
        Default Matplotlib patch keyword arguments used for every aperture.

    Returns
    -------
    list
        Flat list of all patches added to the axes.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    plotted = []
    for aperture in apertures:
        local_kwargs = dict(kwargs)
        patches = aperture.plot(ax=ax, origin=origin, **local_kwargs)
        plotted.extend(to_patch_list(patches))
    return plotted
