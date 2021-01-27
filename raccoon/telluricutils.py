#!/usr/bin/env python
"""
Telluric mask file format:
    - Columns: (0) wavelength [A], (1) "flux".
    - Every region (line or set of lines) that should be blocked because it is affected by tellurics is represented by 0110 in the flux column.
"""
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Constants
C_MS = 2.99792458*1.e8  # Light speed [m/s]
C_KMS = 2.99792458*1.e5  # Light speed [km/s]

###############################################################################


def read_mask(filin):
    w, f = np.loadtxt(filin, usecols=[0, 1], unpack=True)
    return w, f


def mask2wlimits(w, f):
    """Get wavelength limits of the telluric lines masked.
    If a line is given by:
        wa 0
        wb 1
        wc 1
        wd 0
    then get wb as the first limit (store it in `w1`) and wc as the second limit (store it in `w2`).
    """
    w1, w2 = [], []
    for i in range(len(w)-1):
        # Normal line 0110
        if f[i-1] == 0 and f[i] == 1 and f[i+1] == 1 and f[i+2] == 0:
            w1.append(w[i])
            w2.append(w[i+1])
        # Blended lines 01110
        elif f[i-1] == 0 and f[i] == 1 and f[i+1] == 1 and f[i+2] == 1 and f[i+3] == 0:
            w1.append(w[i])
            w2.append(w[i+2])
        # Blended lines 011110
        elif f[i-1] == 0 and f[i] == 1 and f[i+1] == 1 and f[i+2] == 1 and f[i+3] == 1 and f[i+4] == 0:
            w1.append(w[i])
            w2.append(w[i+3])
    w1, w2 = np.asarray(w1), np.asarray(w2)
    return w1, w2


def wlimits2mask(w1, w2, dw=0.001):
    w, f = [], []
    for i in range(len(w1)):  # for each line
        w += [w1[i]-dw, w1[i], w2[i], w2[i]+dw]
        f += [0., 1., 1., 0.]
    w, f = np.asarray(w), np.asarray(f)
    return w, f


def interp_mask_inverse(w, f, kind='linear'):
    """Interpolation function of the telluric mask inverted to be used to mask telluric regions in the data.

    Example
    -------

    >>> # wt, ft : Telluric mask
    >>> # ws, fs : Spectrum
    >>> # Make the inverted mask function
    >>> MaskInv, _ = interp_mask_inverse(wt, ft) # function
    >>> # Make the spectrum flux 0 where there are tellurics
    >>> fs_clean = MaskInv(ws)
    >>> # Plot
    >>> plt.plot(ws, fs, label='spec')
    >>> plt.plot(ws, fs_clean, label='spec clean')
    >>> plt.fill_between(wt, 0, ft, label='tell mask')
    """
    f_inv = np.array(~np.array(f, dtype=bool), dtype=float)
    MaskInv = interp1d(w, f_inv, kind=kind)
    return MaskInv, f_inv


def broaden_mask(w, f, dv):
    """Broaden mask by a velocity `dv` [m/s].

    w_broaden = w*(1+dv/(2.99792458*10**8))

    Parameters
    ----------
    w, f : array
    dv : float
        Velocity to add/subtract to each telluric line, in  m/s.

    Returns
    -------
    w_broaden : array
    """

    C_MS = 2.99792458e8  # [m/s]

    dv = abs(dv)

    w_broaden = [[]]*len(w)
    idone = np.zeros(len(w))
    i = 0
    while i < len(w):
        if idone[i] == 0 and i < len(w)-1:
            # Line (where flux = 1)
            if f[i] == 1 and f[i+1] == 1:
                w_broaden[i] = w[i]*(1-dv/C_MS)
                w_broaden[i+1] = w[i+1]*(1+dv/C_MS)
                idone[i], idone[i+1] = 1, 1
                i = i+2
            # Base of the lines (where flux = 0)
            elif f[i] == 0 and f[i+1] == 0:
                w_broaden[i] = w[i]*(1+dv/C_MS)
                w_broaden[i+1] = w[i+1]*(1-dv/C_MS)
                idone[i], idone[i+1] = 1, 1
                i = i+2
            # Half base (can only happend at the beginning)
            elif f[i] == 0 and f[i+1] == 1:
                w_broaden[i] = w[i]
                i = i+1

        # Last value
        elif idone[i] == 0 and i == len(w)-1:
            if f[i] == 1:  # ...01
                w_broaden[i] = w[i]*(1-dv/C_MS)
            if f[i] == 0:  # ...10
                w_broaden[i] = w[i]*(1+dv/C_MS)
            i = i+1
    w_broaden = np.asarray(w_broaden)
    return w_broaden


def broaden_wlimits(w1, w2, dv):
    """Same as `broaden_mask`, but takes as input the wavelength limits of each mask line, instead of the wavelength and flux (0110).
    """
    C_MS = 2.99792458e8  # [m/s]

    dv = abs(dv)
    w1_broaden = np.asarray(w1)*(1-dv/C_MS)
    w2_broaden = np.asarray(w2)*(1+dv/C_MS)
    return w1_broaden, w2_broaden


def is_there_overlap_wlimits(w1, w2):
    """Check if there is overlap between 2 consecutive telluric regions."""
    ret = False
    for i in range(len(w1)-1):
        if w2[i] >= w1[i+1]:
            ret = True
            break
    return ret


def join_overlap_wlimits_once(w1, w2):
    """Join consecutive telluric regions if there is overlap.
    Only check once.
    """
    w1new, w2new = [], []
    iused = -1
    for i in range(len(w1)-1):  # for each telluric region
        if i == iused:
            continue
        w1new.append(w1[i])
        if w2[i] >= w1[i+1]:  # Join lines
            # w1new.append(w1[i])
            w2new.append(w2[i+1])
            iused = i+1
        else:
            w2new.append(w2[i])

    return w1new, w2new


def join_overlap_wlimits(w1, w2):
    """Join consecutive telluric regions if there is overlap.
    Join until there is no overlap.
    """
    while is_there_overlap_wlimits(w1, w2):
        w1, w2 = join_overlap_wlimits_once(w1, w2)
    return w1, w2


# Plots

def plot_mask(w, f, ax=None, fscale=1., zorder=0, xlab='', ylab='', leglab='', title='', **kwargs):
    if ax is None: ax = plt.gca()
    ax.fill_between(w, f, zorder=zorder, label=leglab, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax
