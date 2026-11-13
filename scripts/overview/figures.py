"""Figures of the project overview page (docs/source/narrative/project_overview.md).

Toy data only (no survey data): each figure illustrates one idea.
Colors follow the talk template: #A11D33 (theme), #641220 (arrows), #434343 (text).
Run from anywhere: python scripts/overview/figures.py -> docs/source/narrative/figures/overview/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, DARK, GRAY, LIGHT = "#A11D33", "#641220", "#434343", "#BDBDBD"
OUT = Path(__file__).resolve().parents[2] / "docs/source/narrative/figures/overview"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 13, "text.color": GRAY, "axes.labelcolor": GRAY,
                     "xtick.color": GRAY, "ytick.color": GRAY, "axes.edgecolor": GRAY})
rng = np.random.default_rng(1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1. a stream
def fig_stream():
    """A satellite on its orbit, stretched into leading and trailing tails."""
    fig, ax = plt.subplots(figsize=(5, 5))
    t = np.linspace(0, 2 * np.pi, 400)
    a, b, rot = 1.0, 0.65, 0.5
    x = a * np.cos(t) * np.cos(rot) - b * np.sin(t) * np.sin(rot)
    y = a * np.cos(t) * np.sin(rot) + b * np.sin(t) * np.cos(rot)
    ax.plot(x, y, ls=":", color=LIGHT, lw=1.5)                       # orbit
    # galaxy: disk + halo glow
    for r, al in [(0.45, 0.06), (0.3, 0.1), (0.15, 0.25)]:
        ax.add_patch(plt.Circle((0, 0), r, color=GRAY, alpha=al, lw=0))
    ax.plot([-0.25, 0.25], [-0.05, 0.05], color=GRAY, lw=4, solid_capstyle="round")
    # debris along the orbit around phase t0, spread in phase and across
    t0 = 1.1
    n = 1500
    dt = rng.normal(0, 0.55, n)
    tt = t0 + dt
    off = rng.normal(0, 0.018, n) * (1 + 2 * np.abs(dt))
    xs = a * np.cos(tt) * np.cos(rot) - b * np.sin(tt) * np.sin(rot)
    ys = a * np.cos(tt) * np.sin(rot) + b * np.sin(tt) * np.cos(rot)
    ax.scatter(xs + off, ys + off, s=2, color=RED, alpha=0.6, lw=0)
    xp = a * np.cos(t0) * np.cos(rot) - b * np.sin(t0) * np.sin(rot)
    yp = a * np.cos(t0) * np.sin(rot) + b * np.sin(t0) * np.cos(rot)
    ax.scatter(xp + rng.normal(0, 0.02, 200), yp + rng.normal(0, 0.02, 200), s=3, color=DARK)
    ax.annotate("satellite\n(cluster or dwarf)", (xp, yp), (xp + 0.25, yp + 0.35),
                color=DARK, ha="left", arrowprops=dict(arrowstyle="-", color=DARK))
    ax.text(0, -0.62, "Milky Way", ha="center", color=GRAY)
    ax.text(-1.05, 0.55, "stream", color=RED, fontweight="bold")
    ax.set_xlim(-1.3, 1.3), ax.set_ylim(-1.3, 1.3), ax.set_aspect("equal"), ax.axis("off")
    save(fig, "1_stream_formation.png")


# ------------------------------------------------------ toy catalog for 2 & 3
def toy_catalog(n_bg=200000, n_st=500):
    """Background stars (broad CMD) + stream stars (on an isochrone, thin band)."""
    # sky, degrees
    xb, yb = rng.uniform(0, 10, n_bg), rng.uniform(0, 10, n_bg)
    xs = rng.uniform(1, 9, n_st)
    ys = 0.6 * xs + 1.8 + rng.normal(0, 0.15, n_st)
    # CMD: background = broad blob, stream = isochrone-like curve
    gb = rng.uniform(17, 24, n_bg)
    cb = 0.25 + 0.08 * (gb - 17) + rng.normal(0, 0.35, n_bg)
    gs = rng.uniform(19.5, 24, n_st)
    iso = lambda g: 0.18 + 0.03 * (g - 19.5) ** 2 - 0.0 * g
    cs = iso(gs) + rng.normal(0, 0.03, n_st)
    return (xb, yb, gb, cb), (xs, ys, gs, cs), iso


def in_filter(g, c, iso, w=0.08):
    return (np.abs(c - iso(g)) < w) & (g > 19.5)


def fig_matched_filter():
    (xb, yb, gb, cb), (xs, ys, gs, cs), iso = toy_catalog()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw=dict(wspace=0.35))
    # all stars on the sky
    x, y = np.r_[xb, xs], np.r_[yb, ys]
    H, *_ = np.histogram2d(x, y, bins=40, range=[[0, 10], [0, 10]])
    axs[0].imshow(H.T, origin="lower", cmap="Greys", extent=[0, 10, 0, 10])
    axs[0].set_title("all stars: nothing visible")
    # CMD with the filter
    axs[1].scatter(cb[:15000], gb[:15000], s=1, color=LIGHT, lw=0)
    axs[1].scatter(cs, gs, s=2, color=RED, lw=0)
    g = np.linspace(19.5, 24, 100)
    axs[1].fill_betweenx(g, iso(g) - 0.08, iso(g) + 0.08, color=RED, alpha=0.15)
    axs[1].plot(iso(g) - 0.08, g, iso(g) + 0.08, g, color=DARK, lw=1)
    axs[1].set_xlim(-0.6, 1.6), axs[1].set_ylim(24.2, 17)
    axs[1].set_xlabel("color"), axs[1].set_ylabel("brightness (mag)")
    axs[1].set_title("keep stars of the stream's\nage and distance")
    # filtered stars on the sky
    kb, ks = in_filter(gb, cb, iso), in_filter(gs, cs, iso)
    x, y = np.r_[xb[kb], xs[ks]], np.r_[yb[kb], ys[ks]]
    H, *_ = np.histogram2d(x, y, bins=40, range=[[0, 10], [0, 10]])
    axs[2].imshow(H.T, origin="lower", cmap="Greys", extent=[0, 10, 0, 10])
    axs[2].set_title("filtered stars: the stream")
    for a in (axs[0], axs[2]):
        a.set_xticks([]), a.set_yticks([]), a.set_xlabel("sky position")
    for x0 in (0.335, 0.635):
        fig.add_artist(FancyArrowPatch((x0, 0.5), (x0 + 0.035, 0.5), transform=fig.transFigure,
                                       arrowstyle="simple,head_width=10,head_length=10", color=DARK))
    save(fig, "2_matched_filter.png")


# -------------------------------------------------------- 3. by-eye bottleneck
def fig_bottleneck():
    """Many maps (distances x sky regions) to inspect by eye; one hides a stream."""
    fig, axs = plt.subplots(3, 6, figsize=(12, 6.2))
    hit = (1, 3)
    for i in range(3):
        for j in range(6):
            m = rng.poisson(20, (60, 60)).astype(float)
            if (i, j) == hit:
                yy, xx = np.mgrid[:60, :60]
                d = np.abs(0.7 * xx - yy + 10) / np.hypot(0.7, 1)
                m += rng.poisson(2.5 * np.exp(-0.5 * (d / 2.0) ** 2))
            from numpy.fft import fft2, ifft2  # light smoothing, as in published maps
            k = np.exp(-0.5 * (np.fft.fftfreq(60)[:, None] ** 2 + np.fft.fftfreq(60)[None] ** 2) / 0.06 ** 2)
            m = np.real(ifft2(fft2(m) * k))
            axs[i, j].imshow(m, cmap="Greys", origin="lower", vmin=18.3, vmax=22.3)
            axs[i, j].set_xticks([]), axs[i, j].set_yticks([])
            if i == 0:
                axs[i, j].set_title(f"{[10, 15, 20, 30, 40, 60][j]} kpc", fontsize=11)
        axs[i, 0].set_ylabel(f"region {i + 1}", fontsize=11)
    for s in axs[hit].spines.values():
        s.set_color(RED), s.set_linewidth(3)
    fig.suptitle("one survey = many maps to inspect by eye.  Which one holds a stream?",
                 color=GRAY, y=0.98)
    save(fig, "3_bottleneck.png")


# ----------------------------------------------------- 4. line sums (Hough)
def line_sums(img, n_theta=90):
    """Sum of an image along every straight line (theta, rho), divided by sqrt(n)."""
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n] - n / 2 + 0.25  # off the half-integers: no empty rho bins
    thetas = np.deg2rad(np.arange(n_theta) * 180 / n_theta)
    rmax = int(np.ceil(n / np.sqrt(2)))
    H = np.full((n_theta, 2 * rmax + 1), np.nan)
    for k, th in enumerate(thetas):
        r = np.round(xx * np.cos(th) + yy * np.sin(th)).astype(int) + rmax
        s = np.bincount(r.ravel(), img.ravel(), 2 * rmax + 1)
        c = np.bincount(r.ravel(), minlength=2 * rmax + 1)
        ok = c >= 20
        H[k, ok] = s[ok] / np.sqrt(c[ok])
    return H, rmax


def fig_line_sums():
    n = 96
    img = rng.normal(0, 1, (n, n))
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    th0, r0 = np.deg2rad(60), 12
    on = np.abs(xx * np.cos(th0) + yy * np.sin(th0) - r0) < 1.0
    img[on] += 0.6                                         # 0.6 sigma per pixel
    H, rmax = line_sums(img)
    k, j = np.unravel_index(np.nanargmax(H), H.shape)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(wspace=0.45))
    axs[0].imshow(img, cmap="Greys", origin="lower")
    axs[0].set_title("a faint stream in noise\n(invisible by eye)")
    axs[0].set_xticks([]), axs[0].set_yticks([])
    axs[1].imshow(H, origin="lower", aspect="auto", cmap="Reds", vmin=1, vmax=np.nanmax(H),
                  extent=[-rmax, rmax, 0, 180])
    axs[1].scatter(j - rmax, k * 2, s=300, facecolor="none", edgecolor=GRAY, lw=2)
    axs[1].set_title(f"sum along every line:\none clear peak ({np.nanmax(H):.1f}σ)")
    axs[1].set_xlabel("line offset"), axs[1].set_ylabel("line angle (°)")
    fig.add_artist(FancyArrowPatch((0.45, 0.5), (0.52, 0.5), transform=fig.transFigure,
                                   arrowstyle="simple,head_width=12,head_length=12", color=DARK))
    save(fig, "4_line_sums.png")


# ----------------------------------------------------- 5. ML pipeline
def fig_pipeline():
    steps = ["simulate\nstreams", "inject into\nthe real sky", "train the\nnetwork",
             "scan the\nwhole survey", "candidate\nstreams"]
    fig, ax = plt.subplots(figsize=(12, 2.2))
    w, h, gap = 1.8, 1.0, 0.55
    for i, s in enumerate(steps):
        x = i * (w + gap)
        last = i == len(steps) - 1
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=RED if last else "white", ec=RED, lw=2))
        ax.text(x + w / 2, h / 2, s, ha="center", va="center",
                color="white" if last else GRAY, fontsize=14, fontweight="bold" if last else None)
        if not last:
            ax.add_patch(FancyArrowPatch((x + w + 0.07, h / 2), (x + w + gap - 0.07, h / 2),
                                         arrowstyle="simple,head_width=8,head_length=8", color=DARK))
    ax.annotate("", (1.5 * (w + gap) + w / 2 - 0.3, -0.25), (0.5 * w, -0.25))
    ax.text(1 * (w + gap) - gap / 2, -0.42, "labels for free: we know where we put the streams",
            ha="center", va="center", fontsize=12, style="italic", color=RED)
    ax.set_xlim(-0.2, len(steps) * (w + gap)), ax.set_ylim(-0.6, 1.2), ax.axis("off")
    save(fig, "5_ml_pipeline.png")


# ----------------------------------------------------- 6. scorecard
def fig_scorecard():
    """The 14 DES 2018 streams (Shipp et al. 2018) and what finds them.

    Numbers from docs/source/experiments/line_model/summary.md (sky search, both
    searches combined: 9 of 14; per-pixel first training: 5 of 14 along the tracks).
    """
    names = ["ATLAS", "Elqui", "Phoenix", "Jhelum", "Chenab", "Tucana III", "Indus",
             "Willka Yaku", "Aliqa Uma", "Wambelong", "Turbio", "Turranburra", "Molonglo", "Ravi"]
    pixel = {"ATLAS", "Elqui", "Chenab", "Tucana III", "Willka Yaku"}
    sky = {"ATLAS", "Elqui", "Phoenix", "Jhelum", "Chenab", "Tucana III", "Indus",
           "Willka Yaku", "Wambelong"}
    absent = {"Molonglo", "Ravi"}
    rows = [("pixel network, along known tracks", pixel),
            ("line network + line sums, blind sky search", sky)]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    for r, (lab, found) in enumerate(rows):
        yv = 1 - r
        for i, nm in enumerate(names):
            if nm in absent:
                ax.scatter(i, yv, s=260, marker="x", color=LIGHT, lw=2)
            else:
                ax.scatter(i, yv, s=260, color=RED if nm in found else "white", ec=RED, lw=1.8)
        ax.text(-0.8, yv, lab, ha="right", va="center")
        ax.text(len(names) - 0.2, yv, f"{len(found)} / 14", va="center", color=RED,
                fontweight="bold", fontsize=15)
    for i, nm in enumerate(names):
        ax.text(i, 1.55, nm, rotation=45, ha="left", va="bottom", fontsize=10)
    ax.text(len(names) - 1.5, -0.65, "×: not in our data", color="#7F7F7F", fontsize=10, ha="center")
    ax.set_xlim(-0.6, len(names) + 0.8), ax.set_ylim(-0.9, 2.6), ax.axis("off")
    save(fig, "6_scorecard.png")


# ----------------------------------------------------- 7. the model
def fig_model():
    """Line network: maps in, one probability per line out (models.hough.HoughUNet)."""
    n = 96
    yy, xx = np.mgrid[:n, :n] - n / 2 + 0.25
    th0, r0 = np.deg2rad(60), 12
    band = np.exp(-0.5 * ((xx * np.cos(th0) + yy * np.sin(th0) - r0) / 1.5) ** 2)
    maps = [rng.normal(0, 1, (n, n)) + a * band for a in (0.35, 0.8, 0.35, 0.0)]
    H, rmax = line_sums(maps[1])
    Hs = np.where(np.isnan(H), 0, H)
    fy, fx = np.fft.fftfreq(Hs.shape[0])[:, None], np.fft.fftfreq(Hs.shape[1])[None]
    Hs = np.real(np.fft.ifft2(np.fft.fft2(Hs) * np.exp(-2 * np.pi ** 2 * 2.0 ** 2 * (fx ** 2 + fy ** 2))))
    Hs[np.isnan(H)] = np.nan
    prob = 1 / (1 + np.exp(-(Hs - np.nanmax(Hs)) * 4 - 3))       # illustration of the output

    fig = plt.figure(figsize=(14, 5.6))
    def box(x, y, w, h, text, fc, ec=GRAY, tc=GRAY, fs=12):
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.015",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=1.5))
        fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs)
    def arrow(x0, x1, y=0.58):
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                       arrowstyle="simple,head_width=9,head_length=9", color=DARK))
    TRAIN, FIXED = "#F2D7DC", "#EDEDED"
    # input: a stack of maps
    labels = ["d − 0.5", "d", "d + 0.5", "off-track\ncolor box"]
    for k, (m, lab) in enumerate(zip(maps[::-1], labels[::-1])):
        ax = fig.add_axes([0.02 + 0.018 * k, 0.40 - 0.035 * k + 0.1, 0.12, 0.30])
        ax.imshow(m, cmap="Greys", origin="lower", vmin=-2.5, vmax=3)
        ax.set_xticks([]), ax.set_yticks([])
    fig.text(0.11, 0.93, "INPUT", ha="center", fontweight="bold", color=RED)
    fig.text(0.11, 0.86, "star-count maps of one sky window", ha="center", fontsize=11)
    fig.text(0.11, 0.36, "matched filter at 3 distances\n+ a background-only map\n(+ the distances: 7 channels)",
             ha="center", va="top", fontsize=10)
    arrow(0.195, 0.225)
    box(0.23, 0.45, 0.13, 0.26, "U-Net\nlocal features\nper pixel\n(+ the input itself)", TRAIN, ec=RED)
    arrow(0.365, 0.395)
    box(0.40, 0.45, 0.13, 0.26, "sum along\nevery line\n(Hough)", FIXED)
    arrow(0.535, 0.565)
    box(0.57, 0.45, 0.13, 0.26, "small CNN\nover the lines", TRAIN, ec=RED)
    arrow(0.705, 0.735)
    fig.text(0.465, 0.93, "MODEL", ha="center", fontweight="bold", color=RED)
    fig.text(0.465, 0.86, "6,210 lines per window: 90 angles × 69 offsets", ha="center", fontsize=11)
    fig.text(0.295, 0.40, "trained", ha="center", fontsize=10, color=RED)
    fig.text(0.465, 0.40, "fixed, not trained", ha="center", fontsize=10)
    fig.text(0.635, 0.40, "trained", ha="center", fontsize=10, color=RED)
    # output: probability per line, and the best line on the sky
    ax = fig.add_axes([0.745, 0.40, 0.11, 0.38])
    ax.imshow(prob, origin="lower", aspect="auto", cmap="Reds", extent=[-rmax, rmax, 0, 180], vmin=0, vmax=1)
    ax.set_xlabel("line offset", fontsize=10), ax.set_ylabel("line angle", fontsize=10, labelpad=2)
    ax.set_xticks([]), ax.set_yticks([])
    ax2 = fig.add_axes([0.875, 0.40, 0.11, 0.38])
    ax2.imshow(maps[1], cmap="Greys", origin="lower", vmin=-2.5, vmax=3)
    t = np.linspace(-60, 60, 2)
    ax2.plot(n / 2 - 0.25 + r0 * np.cos(th0) - t * np.sin(th0), n / 2 - 0.25 + r0 * np.sin(th0) + t * np.cos(th0),
             color=RED, lw=2.5)
    ax2.set_xlim(0, n - 1), ax2.set_ylim(0, n - 1), ax2.set_xticks([]), ax2.set_yticks([])
    fig.text(0.865, 0.93, "OUTPUT", ha="center", fontweight="bold", color=RED)
    fig.text(0.865, 0.86, "probability per line", ha="center", fontsize=11)
    fig.text(0.865, 0.30, "best line = stream track", ha="center", va="top", fontsize=10)
    # training
    box(0.23, 0.05, 0.47, 0.14,
        "training:  label = the lines along the injected stream (yes / no per line)\n"
        "loss = cross-entropy per line  ·  30% of windows hold no stream", "white", ec=RED, fs=11)
    save(fig, "7_model.png")


# ----------------------------------------------------- 8. line length
def fig_length():
    """Idealized line S/N against the stream's visible length (no network).

    A stream adds s (in background-noise units) to each pixel along its track;
    a line of n pixels sums them and divides by sqrt(n). Window lines are about
    96 pixels (11 deg), sub-window (segment) lines 48 pixels (5.5 deg);
    0.115 deg per pixel.
    """
    scale, s = 11 / 96, 0.6
    L = np.linspace(0.5, 15, 300)
    window = s * np.minimum(L, 11) / scale / np.sqrt(96)
    segment = s * np.minimum(L, 5.5) / scale / np.sqrt(48)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(L, window, color=RED, lw=2.5, label="window line (11°)")
    ax.plot(L, segment, color=GRAY, lw=2, ls="--", label="sub-window line (5.5°)")
    ax.axvspan(0, 2, color=LIGHT, alpha=0.4, lw=0)
    ax.text(0.1, 4.6, "shorter\nthan the\nlabel's\nminimum:\ntaught\nas\n\"no line\"",
            ha="left", va="top", fontsize=8)
    for x, lab in [(2, "label min.\nnow (2°)"), (4, "label min.\nbefore (4°)")]:
        ax.axvline(x, color=DARK, lw=1, ls=":")
        ax.text(x + 0.15, 0.3, lab, fontsize=9, color=DARK)
    ax.axvline(11, color=GRAY, lw=1, ls=":")
    ax.text(11.2, 0.3, "longer than the window:\nseen in pieces", fontsize=9)
    ax.set_xlim(0, 15), ax.set_ylim(0, 7)
    ax.set_xlabel("stream length inside the window (°)")
    ax.set_ylabel("line sum (σ)")
    ax.set_title(f"same stream brightness ({s}σ per pixel), different lengths", fontsize=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.18, 1.0), fontsize=10)
    save(fig, "8_line_length.png")


# ----------------------------------------------------- 9. windows -> tracks
def fig_tracks():
    """Half-overlapping windows each report straight segments; segments are joined
    into tracks (directions within 5 deg, gap below 1 deg), one straight
    (great-circle) fit per track, as in run.py line_sky_tracks. Flat toy sky."""
    size, step = 11.0, 5.5
    x = np.linspace(3, 41, 3000)
    y = 12 + 3.5 * np.sin((x - 3) / 38 * np.pi)
    starts = [(x0, y0) for x0 in np.arange(0, 44 - size + 1e-6, step)
              for y0 in np.arange(0, 28 - size + 1e-6, step)]
    segs = []
    for x0, y0 in starts:
        inside = (x >= x0) & (x <= x0 + size) & (y >= y0) & (y <= y0 + size)
        if inside.sum() < 2 or np.hypot(np.ptp(x[inside]), np.ptp(y[inside])) < 2:
            continue
        a_, b_ = np.polyfit(x[inside], y[inside], 1)
        xs = np.array([x[inside].min(), x[inside].max()])
        segs.append(dict(x=xs, y=a_ * xs + b_, ang=np.degrees(np.arctan(a_)), tile=(x0, y0)))
    # union-find linking
    parent = list(range(len(segs)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    def gap(p, q):
        P = np.c_[np.linspace(*p["x"], 50), np.linspace(*p["y"], 50)]
        Q = np.c_[np.linspace(*q["x"], 50), np.linspace(*q["y"], 50)]
        return np.min(np.hypot(*(P[:, None] - Q[None]).transpose(2, 0, 1)))
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if abs(segs[i]["ang"] - segs[j]["ang"]) <= 5 and gap(segs[i], segs[j]) <= 1:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(segs)):
        groups.setdefault(root(i), []).append(i)
    palette = [RED, "#2F6DB5", "#E08A1E", "#3A8F5A", "#7B4FA3", GRAY]

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, gridspec_kw=dict(wspace=0.05))
    ax = axs[0]
    ax.plot(x, y, color=GRAY, lw=7, alpha=0.15)
    t0 = segs[3]["tile"]
    for dx, dy, ls in [(0, 0, "-"), (step, 0, "--")]:
        ax.add_patch(plt.Rectangle((t0[0] + dx, t0[1] + dy), size, size, fill=False, ec=DARK, lw=2, ls=ls))
    ax.text(t0[0] + 0.3, t0[1] + size - 0.4, "a window", va="top", color=DARK, fontsize=10)
    ax.text(t0[0] + step + size, t0[1] - 0.4, "its neighbour, half overlapping",
            va="top", ha="right", color=DARK, fontsize=10)
    for sg in segs:
        ax.plot(sg["x"], sg["y"], color=RED, lw=2)
    ax.set_title(f"each window reports its best straight line\n({len(segs)} segments: one per window crossed)",
                 fontsize=12)
    ax = axs[1]
    ax.plot(x, y, color=GRAY, lw=7, alpha=0.15, label="true stream")
    for k, idx in enumerate(groups.values()):
        c = palette[k % len(palette)]
        X = np.concatenate([segs[i]["x"] for i in idx]); Y = np.concatenate([segs[i]["y"] for i in idx])
        a_, b_ = np.polyfit(X, Y, 1)
        xs = np.array([X.min(), X.max()])
        ax.plot(xs, a_ * xs + b_, color=c, lw=3)
        for i in idx:
            ax.plot(segs[i]["x"], segs[i]["y"], color=c, lw=1, alpha=0.5)
    ax.set_title(f"joined when directions agree within 5° and gap < 1°:\n"
                 f"{len(groups)} tracks, each fitted by one great circle", fontsize=12)
    ax.legend(frameon=False, loc="lower center")
    for a in axs:
        a.set_xlim(0, 44), a.set_ylim(4, 24), a.set_aspect("equal")
        a.set_xticks([]), a.set_yticks([])
    fig.add_artist(FancyArrowPatch((0.495, 0.5), (0.53, 0.5), transform=fig.transFigure,
                                   arrowstyle="simple,head_width=10,head_length=10", color=DARK))
    save(fig, "9_windows_to_tracks.png")

if __name__ == "__main__":
    fig_stream(); fig_matched_filter(); fig_bottleneck()
    fig_line_sums(); fig_pipeline(); fig_scorecard(); fig_model()
    fig_length(); fig_tracks()
    print("figures written to", OUT)
