#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
import matplotlib.pyplot as plt

###############################################################################


class Labeloffset():
    """
    https://stackoverflow.com/questions/45760763/how-to-move-the-y-axis-scale-factor-to-the-position-next-to-the-y-axis-label

    Usage
    -----

    fig, ax = ...
    ...
    Labeloffset(ax, label="my label", axis="y")

    """
    def __init__(self,  ax, label="", axis="y"):
        self.axis = {"y": ax.yaxis, "x": ax.xaxis}[axis]
        self.label = label

        # Format: latex e.g. "10^7" instead of "e7"
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 5))
        self.axis.set_major_formatter(formatter)

        ax.callbacks.connect(axis+'lim_changed', self.update)
        ax.figure.canvas.draw()
        self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + " " + fmt.get_offset())


# Plot styles

colorcycles = {
    # Default matplotlib: blue, orange...
    'vega_category10': ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    # Default matplotlib + black, change orange <-> blue order: black, orange, blue...
    'vega_category10_custom': ['k', "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
}


def mpl_custom_basic():
    """
    Customize basic appearance:
    - ticks in all axis
    - tick direction in, size major 6 minor 3, width 1.2 all
    - axes line width 1.2
    - marker size 7 and line width 2
    - custom colorcycle: black, orange, blue...

    To activate just call this function:
    ```
    import plotutils
    plotutils.mpl_custom_basic()
    ```

    Can call different styles to control different aspects of the plot:
    ```
    plotutils.mpl_custom_basic()
    plotutils.mpl_size_same()
    ```

    To return to matplotlib defaults run:
    ```
    import matplotlib as mpl
    mpl.rcdefaults()
    ```
    """
    mpl.rcParams.update({
        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'xtick.bottom': True,
        'ytick.left': True,
        'ytick.right': True,
        # Tick width
        'xtick.major.width': 1.2,
        'xtick.minor.width': 1.2,
        'ytick.major.width': 1.2,
        'ytick.minor.width': 1.2,
        # Tick size (points)
        'xtick.major.size': 7,
        'xtick.minor.size': 4,
        'ytick.major.size': 7,
        'ytick.minor.size': 4,
        # Axes width
        'axes.linewidth': 1.2,
        # Marker and lines
        'lines.linewidth': 2.0,
        # 'lines.linewidth': 3.0,
        'lines.markersize': 7,
        # 'lines.markeredgewidth': 0, # Problem with '+' and 'x'
        # Colors
        'axes.prop_cycle': mpl.cycler(color=colorcycles['vega_category10_custom']),
        # Axes offset disable
        'axes.formatter.useoffset': True,
        # 'figure.dpi': 200,
        })
    # mpl.rcParams.update({'histtype': 'step'})
    # mpl.rcdefaults()
    return


def mpl_size_same(font_size=18):
    mpl.rcParams.update({
        # General font
        'font.size': font_size,
        # Tick labels
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium',
        # Legend
        'legend.fontsize': 'medium',
        # Axes labels
        'axes.labelsize': 'medium',
        # Title
        'axes.titlesize': 'medium',
        })
    return


# Output

def figout(fig, tl=True, sv=True, filout='', svext=['png', 'pdf'], sh=False, cl=True):
    if tl: fig.tight_layout()
    if sv:
        for ext in svext: fig.savefig(filout+'.'+ext)
    if sh: plt.show()
    if cl: plt.close(fig)
    return


def figout_simple(fig, sv=True, filout='', svext=['png', 'pdf'], sh=False, cl=True, rasterized=True):
    """
    """
    if sv:
        for ext in svext: fig.savefig(filout+'.'+ext, rasterized=rasterized)
    if sh: plt.show()
    if cl: plt.close(fig)
    return
