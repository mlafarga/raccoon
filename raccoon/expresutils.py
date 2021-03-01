#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import glob
import os
import sys

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from . import fitsutils

###############################################################################


# -----------------------------------------------------------------------------
#
# DRS utils
#
# -----------------------------------------------------------------------------

# FITS
# ----

# Spectra and header

def drs_fitsred_read(filin):
    """
    Read reduced spectrum FITS.
    """
    # Read FITS
    with fits.open(filin) as hdulist:
        # Headers (0 main, 1 activity ind, 2 low-res chromatic exposure meter spectrograph and barycentric correction)
        header0 = hdulist[0].header
        header1 = hdulist[1].header
        header2 = hdulist[2].header

        # Optimally extracted data: extension 1
        w = hdulist[1].data['wavelength']
        wcb = hdulist[1].data['bary_wavelength']  # chromatic-barycentric corrected w
        we = hdulist[1].data['excalibur']  # excalibur w
        wecb = hdulist[1].data['bary_excalibur']  # chromatic-barycentric corrected, excalibur w
        mwecb = hdulist[1].data['excalibur_mask']  # mask pix w/o excalibur w (are NaN in array `we` or `wecb`)

        f = hdulist[1].data['spectrum']  # flat-relative optimal extraction
        sf = hdulist[1].data['uncertainty']
        c = hdulist[1].data['continuum']
        b = hdulist[1].data['blaze']  # original counts -> S/N
        mf = hdulist[1].data['pixel_mask']  # mask low-signal pix (order extremes, are NaN in array `f`)

    return w, wcb, we, wecb, mwecb, f, sf, c, b, mf, header0, header1, header2
    # if out == 'all':
    #     return w, wcb, we, wecb, mwecb, f, sf, c, b, mf, header0, header1, header2
    # elif out == 'standard':
    #     return w, f, sf, c, b, header0, header1, header2
    # elif out == 'standardbary':
    #     return wcb, f, sf, c, b, header0, header1, header2
    # elif out == 'excalibur':
    #     return we, f, sf, c, b, header0, header1, header2
    # elif out == 'excaliburbary':
    #     return wecb, f, sf, c, b, mwecb, header0, header1, header2


# Merged spectra

def drs_fitsredmerged_read(filin):
    """
    Read telluric model from merged reduced spectrum FITS.
    """
    # Read FITS
    with fits.open(filin) as hdulist:

        # Headers (0 main, 1 activity ind, 2 low-res chromatic exposure meter spectrograph and barycentric correction)
        header0 = hdulist[0].header
        header1 = hdulist[1].header
        header2 = hdulist[2].header

        # Telluric model
        w = hdulist[1].data['bary_wavelength']  # uniform in log space
        f = hdulist[1].data['spectrum']
        fe = hdulist[1].data['spectrum_excalibur']

        # B-spline information
        knots = hdulist[2].data['spectrum_knots']
        coeffs = hdulist[2].data['spectrum_coeffs']
        knotse = hdulist[2].data['spectrum_excalibur_knots']
        coeffse = hdulist[2].data['spectrum_excalibur_coeffs']
        degree = hdulist[0].header['BDEGREE']  # Kind of a curve ball

        # Generate B-spline Function
        #   Mask zeros at the end of the arrays. If not, get "ValueError: Knots must be in a non-decreasing order."
        maskzero = knots != 0
        maskzeroe = knotse != 0
        Tmerged = BSpline(knots[maskzero], coeffs[maskzero], degree, extrapolate=False)
        Tmergede = BSpline(knotse[maskzeroe], coeffse[maskzeroe], degree, extrapolate=False)

    return w, f, fe, knots, coeffs, knotse, coeffse, degree, Tmerged, Tmergede, header0, header1, header2


def drs_fitsred_tell_read(filin):
    """
    Read telluric model from merged reduced spectrum FITS.
    """
    # Read FITS
    with fits.open(filin) as hdulist:

        # Headers (0 main, 1 activity ind, 2 low-res chromatic exposure meter spectrograph and barycentric correction)
        header0 = hdulist[0].header
        header1 = hdulist[1].header
        header2 = hdulist[2].header

        # # Optimally extracted data: extension 1
        # w = hdulist[1].data['wavelength']
        # wcb = hdulist[1].data['bary_wavelength']  # chromatic-barycentric corrected w
        # we = hdulist[1].data['excalibur']  # excalibur w
        # wecb = hdulist[1].data['bary_excalibur']  # chromatic-barycentric corrected, excalibur w
        # mwecb = hdulist[1].data['excalibur_mask']  # mask pix w/o excalibur w (are NaN in array `we` or `wecb`)
        w = hdulist[1].data['bary_wavelength']  # chromatic-barycentric corrected w,   # uniform in log space
        # SELENITE-constructed telluric model
        t = hdulist[1].data['tellurics']

    # return w, wcb, we, wecb, mwecb, t, header0, header1, header2
    return w, t, header0, header1, header2


def drs_fitsredmerged_tell_read(filin):
    """
    Read telluric model from merged reduced spectrum FITS.
    """
    # Read FITS
    with fits.open(filin) as hdulist:

        # Headers (0 main, 1 activity ind, 2 low-res chromatic exposure meter spectrograph and barycentric correction)
        header0 = hdulist[0].header
        header1 = hdulist[1].header
        header2 = hdulist[2].header

        # Telluric model
        w = hdulist[1].data['bary_wavelength']  # uniform in log space
        t = hdulist[1].data['tellurics']
        te = hdulist[1].data['tellurics_excalibur']

        # B-spline information
        knots = hdulist[2].data['tellurics_knots']
        coeffs = hdulist[2].data['tellurics_coeffs']
        knotse = hdulist[2].data['tellurics_excalibur_knots']
        coeffse = hdulist[2].data['tellurics_excalibur_coeffs']
        degree = hdulist[0].header['BDEGREE']  # Kind of a curve ball

        # Generate B-spline Function
        #   Mask zeros at the end of the arrays. If not, get "ValueError: Knots must be in a non-decreasing order."
        maskzero = knots != 0
        maskzeroe = knotse != 0
        Tmerged = BSpline(knots[maskzero], coeffs[maskzero], degree, extrapolate=False)
        Tmergede = BSpline(knotse[maskzeroe], coeffse[maskzeroe], degree, extrapolate=False)

    return w, t, te, knots, coeffs, knotse, coeffse, degree, Tmerged, Tmergede, header0, header1, header2



# SNR

def drs_snr(filin, ords):
    """
    Compute SNR per order.
    """
    # Read FITS
    with fits.open(filin) as hdulist:

        # Optimally extracted data: extension 1
        f = hdulist[1].data['spectrum']  # flat-relative optimal extraction
        b = hdulist[1].data['blaze']  # original counts -> S/N
        # mf = hdulist[1].data['pixel_mask']  # mask low-signal pix (order extremes, are NaN in array `f`)

    lissnr = {}
    for o in ords:
        # lissnr[o] = np.sqrt(np.nanmean(f[o] * b[o]))
        lissnr[o] = np.nanmean(np.sqrt(f[o] * b[o]))
    # lissnr = pd.DataFrame.from_dict(lissnr, orient='index', columns=[filin])
    return lissnr


def drs_snr_lisobs(lisobs, ords, name='snro'):
    lissnr = {}
    for obs in lisobs:
        lissnr[obs] = drs_snr(obs, ords)
    lissnr = pd.DataFrame.from_dict(lissnr, orient='index')

    # Change column names
    if name is not None:
        if name == 'snro':
            changecol = {i: 'snro{:d}'.format(i) for i in lissnr.columns}
        lissnr.rename(columns=changecol, inplace=True)

    return lissnr


# More header data

def drs_bjd_lisobs(lisobs, notfound=np.nan, ext=2, name='bjd'):
    """Get MJD from FITS header ('HIERARCH wtd_mdpt') for the observations in `lisobs`. Add  + 2400000.5 to value in the header to pass from MJD to BJD.

    Returns
    -------
    data : pandas dataframe
    """
    kw = 'HIERARCH wtd_mdpt'
    if name is not None: names = {kw: name}
    else: names = None
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=names)
    if name is not None: data[name] = data[name] # + 2400000.
    else: data[kw] = data[kw] # + 2400000.
    return data


def drs_exptime_lisobs(lisobs, notfound=np.nan, ext=0, name='exptime'):
    """Get the readout noise from FITS header ('AEXPTIME') [s] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kw = 'AEXPTIME'
    if name is not None: names = {kw: name}
    else: names = None
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=names)
    return data


def drs_airmass_lisobs(lisobs, notfound=np.nan, ext=0, name='airmass'):
    """Get the airmass from FITS header ('AIRMASS') [s] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kw = 'AIRMASS'
    if name is not None: names = {kw: name}
    else: names = None
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=names)
    return data


def drs_moondist_lisobs(lisobs, notfound=np.nan, ext=0, name='moondist'):
    """Get the airmass from FITS header ('AIRMASS') [s] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kw = 'MOONDIST'
    if name is not None: names = {kw: name}
    else: names = None
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=names)
    # Transform to float
    data = data.astype(float)
    return data


def apply_expres_masks_array(x, pixel_mask, excalibur_mask=None):
    """
    x : 2d array (ords, pix) (wavelength, or flux...)
    """
    nord = len(x)
    ords = np.arange(0, nord, 1)
    xnew = [[]]*nord
    for o in ords:
        if excalibur_mask is not None:
            mpix = pixel_mask[o] & excalibur_mask[o]
        else:
            mpix = pixel_mask[o]
        xnew[o] = x[o][mpix]
    return xnew


def apply_expres_masks_spec(w, f, sf, c, pixel_mask, excalibur_mask=None):
    nord = len(w)
    ords = np.arange(0, nord, 1)
    wnew, fnew, sfnew, cnew = [[]]*nord, [[]]*nord, [[]]*nord, [[]]*nord
    for o in ords:
        if excalibur_mask is not None:
            mpix = pixel_mask[o] & excalibur_mask[o]
        else:
            mpix = pixel_mask[o]
        wnew[o] = w[o][mpix]
        fnew[o] = f[o][mpix]
        sfnew[o] = sf[o][mpix]
        cnew[o] = c[o][mpix]
    return wnew, fnew, sfnew, cnew
