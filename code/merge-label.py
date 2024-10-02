import numpy as np
from matplotlib import pyplot as pl
from scipy.ndimage import label
from matplotlib.colors import ListedColormap


def translate(p, tile, ul):
    q = tile.copy()
    m = ul.max() + 1
    q += m
    for u in ul:
        if u == -1:
            continue
        vl, lc = np.unique(q[p == u], return_counts=True)
        if len(lc) == 0:
            continue
        v = vl[np.argmax(lc)]
        q[q == v] = u
    vl = np.unique(q)
    vl = vl[vl >= m]
    for v in vl:
        q[q == v] = m
        m += 1
    return q


# make data
X, Y = np.meshgrid(np.linspace(-3, 3, 190), np.linspace(-3, 3, 190))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
Z = (Z*Z > 0.02).astype("int")
Z[(X+2)**2+(Y+2)**2 < 0.4] = 1
Z[(X-2)**2+(Y-2)**2 < 0.3] = 1

T0, _ = label(Z[-100:, :100])
T1, _ = label(Z[-100:, -100:])
T2, _ = label(Z[:100, :100])
T3, _ = label(Z[:100, -100:])

# plot
cmap = ListedColormap(["lavender", "coral", "orange", "gold", "cyan", "lightseagreen"])

fg, ax = pl.subplots(2, 2, sharex=True, sharey=True)
ax = ax.ravel()

im = ax[0].imshow(T0+1, vmin=0.5, vmax=6.5, origin='lower', interpolation="none", cmap=cmap)
im = ax[1].imshow(T1+1, vmin=0.5, vmax=6.5, origin='lower', interpolation="none", cmap=cmap)
im = ax[2].imshow(T2+1, vmin=0.5, vmax=6.5, origin='lower', interpolation="none", cmap=cmap)
im = ax[3].imshow(T3+1, vmin=0.5, vmax=6.5, origin='lower', interpolation="none", cmap=cmap)
fg.colorbar(im, ax=ax, ticks=(1, 2, 3, 4, 5, 6)).set_label("tile specific labels")
pl.show()

# naive wrong way of merging labels
wrong = np.zeros((190, 190))
wrong[-100:, -100:] = T1
wrong[:100, -100:] = T3
wrong[-100:, :100] = T0
wrong[:100, :100] = T2

fg, ax = pl.subplots()
im = ax.imshow(wrong+1, vmin=0.5, vmax=6.5, origin='lower', interpolation="none", cmap=cmap)
fg.colorbar(im, ax=ax, ticks=(1, 2, 3, 4, 5, 6)).set_label("tile specific labels")

pl.show()

# hopefully correct way of merging labels
right = np.zeros((190, 190), dtype="int") - 1

# start with first tile as it is
right[-100:, :100] = T0

# the next tile needs to adapt to what's already there
right[:100, :100] = translate(right[:100, :100], T2, np.unique(right))

# and so on
right[-100:, -100:] = translate(right[-100:, -100:], T1, np.unique(right))
right[:100, -100:] = translate(right[:100, -100:], T3, np.unique(right))

# final result
fg, ax = pl.subplots()
im = ax.imshow(right+1, vmin=0.5, vmax=6.5, origin='lower', interpolation="none", cmap=cmap)
fg.colorbar(im, ax=ax, ticks=(1, 2, 3, 4, 5, 6)).set_label("unified labels")

pl.show()
