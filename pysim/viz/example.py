import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-talk"])

from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder


ref_pt = 10.0
ref_label = "Reference Point"

angle_ax_label = r"Correlation"
x_ax_label = r"Standard Deviation"

# Extend for Standard Deviation
prnt_ext = (0, 1.25)
smin = prnt_ext[0] * ref_pt
smax = prnt_ext[1] * ref_pt

# polar axis transform
tr = PolarAxes.PolarTransform()

# Correlation Labels
rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

# TODO - extend to negative correlations
tmax = np.pi / 2.0

# convert correlation labels to polar angles
tlocs = np.arccos(rlocs)

# positions on grid
gl1 = grid_finder.FixedLocator(tlocs)

# create ticks
tf1 = grid_finder.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

ghelper = floating_axes.GridHelperCurveLinear(
    aux_trans=tr, extremes=(0, tmax, smin, smax), grid_locator1=gl1, tick_formatter1=tf1
)
print(tr)
print((0, tmax, smin, smax))
print(gl1)
print(tf1)
plt.style.use(["seaborn-poster"])

fig = plt.figure(figsize=(8, 8))

subplot = 111
print(fig)
print(subplot)
print(ghelper)
ax = floating_axes.FloatingSubplot(fig, subplot, grid_helper=ghelper)

fig.add_subplot(ax)

plt.show()

