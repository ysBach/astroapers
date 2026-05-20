from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers


def test_plot_returns_matplotlib_artist():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    artist = apers.EllipAp((3.0, 4.0), a=2.0, b=1.0, theta=0.2).plot(ax=ax)
    assert artist.axes is ax


def test_to_patches_does_not_add_artist_to_axes():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    artist = apers.CircAp((3.0, 4.0), r=2.0).to_patches(color="tab:red")

    assert artist.axes is None
    assert artist not in ax.patches


def test_vector_plot_returns_patch_list_and_adds_all_artists():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    positions = np.array([[3.0, 4.0], [8.0, 9.0]])
    artists = apers.CircAp(positions, r=2.0).plot(ax=ax)

    assert isinstance(artists, list)
    assert len(artists) == 2
    assert all(artist.axes is ax for artist in artists)


def test_plot_many_returns_flat_patch_list():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from astroapers.plotting import plot_many

    _, ax = plt.subplots()
    patches = plot_many(
        [
            apers.CircAp((3.0, 4.0), r=2.0),
            apers.RectAp([(10.0, 12.0), (13.0, 14.0)], w=3.0, h=2.0),
        ],
        ax=ax,
    )

    assert len(patches) == 3
    assert all(patch.axes is ax for patch in patches)


def test_rect_plot_vertices_are_in_data_coordinates():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    artist = apers.RectAp((10.0, 20.0), w=8.0, h=4.0).plot(ax=ax)
    vertices = artist.get_xy()[:-1]

    assert_allclose(vertices[:, 0].min(), 6.0)
    assert_allclose(vertices[:, 0].max(), 14.0)
    assert_allclose(vertices[:, 1].min(), 18.0)
    assert_allclose(vertices[:, 1].max(), 22.0)


def test_plot_origin_shifts_cutout_coordinates():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    circle = apers.CircAp((30.0, 40.0), r=5.0).plot(ax=ax, origin=(20.0, 30.0))
    ellipse = apers.EllipAp((30.0, 40.0), a=5.0, b=2.0).plot(ax=ax, origin=(20.0, 30.0))
    rect = apers.RectAp((30.0, 40.0), w=8.0, h=4.0).plot(ax=ax, origin=(20.0, 30.0))

    assert_allclose(circle.center, (10.0, 10.0))
    assert_allclose(ellipse.center, (10.0, 10.0))
    assert_allclose(rect.get_xy()[:-1].mean(axis=0), (10.0, 10.0))


def test_circular_annulus_plot_preserves_inner_curve_codes():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    artist = apers.CircAn((10.0, 20.0), r_in=4.0, r_out=8.0).plot(ax=ax)
    codes = artist.get_path().codes

    assert np.count_nonzero(codes == mpath.Path.MOVETO) == 2
    assert np.count_nonzero(codes == mpath.Path.CURVE4) >= 48


def test_pill_plot_samples_are_tunable():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    aperture = apers.PillAp((3.0, 4.0), w=5.0, a=1.5, b=1.0, plot_samples=32)
    artist = aperture.plot(ax=ax, plot_samples=64)

    assert len(artist.get_path().vertices) == 64
    vertices = artist.get_path().vertices
    assert_allclose(vertices[:, 0].min(), 3.0 - 0.5 * 5.0 - 1.5, atol=5e-3)
    assert_allclose(vertices[:, 0].max(), 3.0 + 0.5 * 5.0 + 1.5, atol=5e-3)
    assert_allclose(vertices[:, 1].min(), 4.0 - 1.0, atol=5e-3)
    assert_allclose(vertices[:, 1].max(), 4.0 + 1.0, atol=5e-3)
    with pytest.raises(ValueError, match="plot_samples"):
        aperture.plot(ax=ax, plot_samples=63)
    with pytest.raises(ValueError, match="plot_samples"):
        apers.PillAp((3.0, 4.0), w=5.0, a=1.5, b=1.0, plot_samples=64.0)


def test_pill_annulus_plot_returns_matplotlib_artist():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    artist = apers.PillAn(
        (3.0, 4.0), w_in=3.0, a_in=1.0, b_in=0.7, w_out=5.0, a_out=1.5, b_out=1.0
    ).plot(ax=ax, plot_samples=64)
    assert artist.axes is ax
