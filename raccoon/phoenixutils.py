#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

from astropy.io import fits
import numpy as np

###############################################################################


# Read

def read_phoenixfits(filw, filf):
    """Read PHOENIX spectrum from FITS files."""
    # Read wavelength
    with fits.open(filw) as hdulist:
        w = hdulist[0].data
    # Read flux
    with fits.open(filf) as hdulist:
        f = hdulist[0].data
    return w, f

# -----------------------------------------------------------------------------


# Process

def slice_phoenix(w1, w2, w, f, verb=False):
    """Slice PHOENIX spectrum from `w1` to `w2`.
    """
    # Indices of the `w` values closest to w1 and w2
    i1 = (np.abs(w-w1)).argmin()
    i2 = (np.abs(w-w2)).argmin()
    if verb:
        print('Slice range: [{},{}]'.format(w1, w2))
        print(' w1={}, wclosest={}, i1={}'.format(w1, w[i1], i1))
        print(' w2={}, wclosest={}, i2={}'.format(w2, w[i2], i2))
    # Slice spectrum
    wcut = w[i1:i2+1]
    fcut = f[i1:i2+1]
    return wcut, fcut


def wvac2air(w):
    """Transform vacuum wavelength to air wavelength.
    Husser 2013, Cidor 1996.
    """
    s2 = (1e4/w)**2
    f = 1.+0.05792105/(238.0185-s2)+0.00167917/(57.362-s2)
    wair = w/f
    return wair

# -----------------------------------------------------------------------------


# Save

def save_phoenixdat(w, f, filout, verb=False):
    """Save spectrum in .dat file
    """
    np.savetxt(filout, zip(w, f), fmt=('%.3f', '%.0f'))
    if verb:
        print('Spectrum saved in', filout)
    return
