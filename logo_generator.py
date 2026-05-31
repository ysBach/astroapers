import astroapers as aap
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


def generate_astroapers_logo(output_filename="astroapers_logo.png"):
    """
    Generates a logo for the astroapers package using exact pixel-aperture overlaps.
    """

    """
    Generates a logo for the astroapers package using exact pixel-aperture overlaps.
    """
    # Color Palette
    rust_orange = "#CE412B"
    python_blue = "#4B8BBE"
    starlight_yellow = "#FFE873"
    grid_color = "#FFFFFF"

    # 1. Setup Figure
    fig_size = 6
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)

    bg_color = "#0D1117"  # GitHub Dark / Deep Space
    fig.patch.set_facecolor(bg_color)
    # ax.set_facecolor(bg_color)

    npix = 25
    image = np.zeros((npix, npix))
    yy, xx = np.mgrid[0:npix, 0:npix]
    # add simple background:
    # image += 0.01*xx + 0.01*yy

    # add a gaussian star:
    xcen = npix // 2
    ycen = npix // 2 + 2
    image += np.exp(-((xx - xcen) ** 2 + (yy - ycen) ** 2) / (2 * (npix / 25) ** 2))
    # add Sersic-like galaxy:
    xcen_gal = npix // 2 + 8
    ycen_gal = npix // 2 + 5
    theta = np.radians(45)  # rotation angle
    x_rot = (xx - xcen_gal) * np.cos(theta) + (yy - ycen_gal) * np.sin(theta)
    y_rot = -(xx - xcen_gal) * np.sin(theta) + (yy - ycen_gal) * np.cos(theta)
    image += 0.8 * np.exp(-1 * (np.sqrt(x_rot**2 + 0.2 * y_rot**2)) ** (1 / 0.5))

    # add a cosmic-ray hit:
    image[npix - np.arange(3, 6), np.arange(3, 6)] += 0.99

    ap = aap.CircAp((xcen + 0.5, ycen + 0.5), r=3)
    an = aap.CircAn((xcen + 0.5, ycen + 0.5), r_in=6, r_out=10)
    ax.imshow(
        image, extent=(0, npix, 0, npix), origin="lower", cmap="gist_heat", zorder=1
    )
    ap.plot(ax, edgecolor=starlight_yellow, facecolor="none", linewidth=2, zorder=3)
    an.plot(
        ax,
        edgecolor="w",
        fill=True,
        facecolor="w",
        alpha=0.25,
        linewidth=1,
        zorder=2,
        ls="--",
    )
    an.plot(ax, edgecolor="w", linewidth=1, zorder=2, ls="--")

    # ap2 = aap.EllipAp((xcen_gal + 0.5, ycen_gal + 0.5), a=4, b=2, theta=theta+np.pi/2)
    # an2 = aap.EllipAn((xcen_gal + 0.5, ycen_gal + 0.5), a_in=7, b_in=4, a_out=10, b_out=6, theta_in=theta+np.pi/2)

    ap3 = aap.PillAp(
        (xcen_gal + 0.5, ycen_gal + 0.5), a=2, b=2, w=4, theta=theta + np.pi / 2
    )
    an3 = aap.PillAn(
        (xcen_gal + 0.5, ycen_gal + 0.5),
        a_in=3,
        b_in=3,
        w_in=4,
        a_out=5,
        b_out=5,
        w_out=7,
        theta_in=theta + np.pi / 2,
    )

    # ap2.plot(ax, edgecolor=python_blue, facecolor='none', linewidth=2, zorder=10)
    # an2.plot(ax, edgecolor=starlight_yellow, fill=True, facecolor=starlight_yellow, alpha=0.2, linewidth=1, ls="--")
    # an2.plot(ax, edgecolor=starlight_yellow, linewidth=1, ls=":")

    ap3.plot(ax, edgecolor=starlight_yellow, facecolor="none", linewidth=2, zorder=10)
    an3.plot(
        ax,
        edgecolor=python_blue,
        fill=True,
        facecolor=python_blue,
        alpha=0.2,
        linewidth=1,
        ls="--",
    )
    an3.plot(ax, edgecolor=python_blue, linewidth=1, ls="--")

    ax.axis("off")
    ax.set_aspect("equal")

    for val in range(0, npix + 1, 1):
        ax.axhline(val, color=grid_color, lw=0.5, alpha=0.3, zorder=1)
        ax.axvline(val, color=grid_color, lw=0.5, alpha=0.3, zorder=1)

    ax.set(xlim=(2, npix - 2), ylim=(4, npix))
    plt.tight_layout()
    plt.savefig(
        "logo.png", dpi=300, bbox_inches="tight", pad_inches=0.0, transparent=True
    )

    plt.close(fig)

    # === Text

    # 1. Setup Figure
    fig_size = (3.4, 2)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    bg_color = "#0D1117"  # GitHub Dark / Deep Space
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.plot([0, fig_size[0]], [0, fig_size[1]], marker="", ls="")

    # add texts:
    font_kw = {
        "fontsize": 60,
        "fontweight": "bold",
        "fontfamily": "monospace",
        "zorder": 10,
    }

    scale = 0.05
    an_o = aap.PillAn(
        (fig_size[0] * 0.9, fig_size[1] / 2 + 0.55),
        a_in=3 * scale,
        b_in=3 * scale,
        w_in=2.5 * scale,
        a_out=5 * scale,
        b_out=5.5 * scale,
        w_out=5 * scale,
        theta_in=np.pi / 2,
    )
    # ap_r = aap.RectAp(
    #     (fig_size[0]*0.67, fig_size[1]/2 + 0.55),
    #     w=4*scale, h=13*scale, theta=0
    # )
    ap_rs = aap.RectAp(
        (fig_size[0] * 0.8, fig_size[1] / 2 - 0.55), w=25 * scale, h=15 * scale, theta=0
    )
    an_o.plot(
        ax, edgecolor=rust_orange, fill=True, facecolor=starlight_yellow, ls="-", lw=1
    )
    # ap_r.plot(ax, edgecolor=rust_orange, fill=True, facecolor=starlight_yellow, ls="", lw=1, zorder=11)
    ap_rs.plot(
        ax,
        edgecolor=rust_orange,
        fill=True,
        facecolor=rust_orange,
        ls="",
        lw=1,
        zorder=9,
        alpha=0.4,
    )

    ax.text(
        0.0,
        fig_size[1] / 2,
        "astr",
        **font_kw,
        color="w",
        ha="left",
        va="bottom",
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground=rust_orange),
            path_effects.Normal(),
        ],
    )
    ax.text(
        0.0,
        fig_size[1] / 2,
        "apers",
        **font_kw,
        color="w",
        ha="left",
        va="top",
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground=python_blue),
            path_effects.Normal(),
        ],
    )

    # rect = aap.RectAp((19.9, 2.75), w=3.5, h=2.1, theta=0)
    # rect.plot(ax, edgecolor=rust_orange, facecolor=rust_orange, fill=True, linewidth=2, zorder=1, alpha=0.5)

    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    # plt.savefig("logo_text.svg", format='svg', bbox_inches='tight', pad_inches=0)
    plt.savefig(
        "logo_text.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )


if __name__ == "__main__":
    generate_astroapers_logo()
