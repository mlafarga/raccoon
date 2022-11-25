#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import glob
import os

from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

import ipdb

import colorcet as cc
import cmocean


from . import peakutils
from . import plotutils
from . import pyutils

# import ccflibfort77 as ccflibfort
# from . import ccflibfort as ccflibfort
from . import ccflibfort
# from raccoon import ccflibfort77 as ccflibfort
# distro = pyutils.get_distro()
# if 'ubuntu' in distro: from raccoon import ccflibfort77ubuntu as ccflibfort
# elif 'debian' in distro: from raccoon import ccflibfort77debian as ccflibfort
# else: import ccflibfort77 as ccflibfort

# dirhere = os.path.dirname(os.path.realpath(__file__))
dirhere = os.path.abspath(os.path.dirname(__file__))
# import pkg_resources
# # dirhere = pkg_resources.resource_filename(__name__, 'data/mask/CARM_VIS/')
# dirhere = pkg_resources.resource_filename('raccoon', 'data/')

###############################################################################

# CCF


def computeccf(w, f, c, wm, fm, rv, ron=None, forig=None):
    """
    Wrapper for the fortran functions `ccflibfort.computeccf`, `ccflibfort.computeccferr`, `computeccferrfluxorig`.

    Compute the CCF error only if `ron` is not None (default). Not computing the error makes the execution slightly faster.

    Parameters
    ----------
    w, f : 1d array like, same shape
        Spectrum (1d, i.e. for 1 order). Wavelength units must be consistent with mask wavelength `wm`.
    c : 1d array like, same shape as `w` and `f`
        Blaze function, to take into account the observed flux. If no blaze, input an array with ones, e.g. `c = np.ones_like(w)`.
    wm, fm : 1d array like, same shape
        Mask position (`wm`) and weight (`fm`).
    rv : 1d array-like
        RV array for which the mask is Doppler-shifted.
    ron : float
        Read out noise. If None (default), no CCF error is computed.
    forig : 1d array
        Original spectrum flux (before correcting for order ratios).

    Returns
    -------
    ccf, ccferr : 1d array
    """

    # Check non-empy arrays
    if len(w) == 0 or len(f) == 0 or len(c) == 0 or len(wm) == 0 or len(fm) == 0:
        ccf, ccferr = rv * np.nan, rv * np.nan
        return ccf, ccferr

    if ron is None:
        ccf = ccflibfort.computeccf(w, f, c, wm, fm, rv)
        ccferr = np.ones_like(ccf) * np.nan

    else:
        if forig is None:
            ccf, ccferr = ccflibfort.computeccferr(w, f, c, wm, fm, rv, ron)
        else:
            ccf, ccferr = ccflibfort.computeccferrfluxorig(w, f, c, forig, wm, fm, rv, ron)
    return ccf, ccferr


def computerverr(rv, ccf, ccferr, returnall=False):
    """Compute RV error from CCF profile.
    """
    der = np.ones_like(rv) * np.nan
    # rverr = np.ones_like(rv) * np.nan
    rverrsum = 0.
    for i in range(len(rv)):

        # Derivative
        if i == 0:  # start
            der[i] = np.abs((ccf[i+1]-ccf[i])/(rv[i+1]-rv[i]))
        elif i == len(rv)-1:  # end
            der[i] = np.abs((ccf[i]-ccf[i-1])/(rv[i]-rv[i-1]))
        else:  # rest
            der[i] = np.abs(((ccf[i+1]-ccf[i])/(rv[i+1]-rv[i]) + (ccf[i]-ccf[i-1])/(rv[i]-rv[i-1]))/2.)

    ccferr = np.array(ccferr)
    der = np.array(der)

    # RV err
    rverr = ccferr / der

    # RV err total
    rverrsum = np.sum(1. / rverr**2)
    rverrt = 1./np.sqrt(rverrsum)
    if returnall:
        return rverrt, der, rverr
    return rverrt


def sumccf(rv, ccf, ords):
    """
    Sum CCFs of different orders.

    Parameters
    ----------
    rv :
    ccf : list of 1d arrays
        Arrays with the CCF of each order.
    ords : array
        Spectrum orders to use.
    """
    ccfsum = np.zeros_like(rv)
    for o in ords:
        ccfsum += ccf[o]
    return ccfsum

###############################################################################


# Fit function

def fitgaussian(x, y):
    """
    Gaussian G(x)=shift+amp*e^(-(x-cen)^2/(2*wid^2))
    """
    lmfitresult = peakutils.fit_gaussian_peak(x, y, amp_hint=-0.5, cen_hint=None, wid_hint=0.01, shift_hint=0.8, minmax='min')
    return lmfitresult


def fitgaussianfortran(rv, ccf, fitrng, funcfitnam='gaussian', nfitpar=4):
    """Fit a Gaussian function to the CCF and return the best fit parameters.

    Uses the subroutine `fitccf` from the fortran library `ccflibfort`.

    Gaussian definition:
        shift + amp * e**(-(x-cen)**2 / wid)
    FWHM = 2*sqrt(ln(2)*wid)

    Parameters
    ----------
    rv, ccf : 1d arrays
    funcfitnam : str, {'gaussian'}
    nfitpar : int, {4}
        Number of parameters to fit (4 in case of a Gaussian)

    Returns
    -------
    fitpar : dict
        Dictionary with the best fit parameters of the Gaussian.

    """
    # Fit parameters and errors
    a, da = [np.nan]*nfitpar, [np.nan]*nfitpar

    # Arbitrary sig
    sig = np.array([.01]*len(rv))  # #### TEST CHANGE VALUES

    # Parameter guess
    ccfmin = min(ccf)  # CCF minimum
    rvccfmin = rv[np.argmin(ccf)]  # RV where CCF is minimum
    ain = np.array([ccfmin-1, rvccfmin, 5., 1.])  # Initial fit params [amp,cen,wid,shift]
    lista = np.array([1, 1, 1, 1], dtype='int')  # 1=fit, 0=no fit

    # # Fit only if CCF minimum near `rv` center
    # rvcen = rv[int(len(rv)/2.)]
    # if rvccfmin<rvcen+10. and rvccfmin>rvcen-10.:
    #    if fitrng=='max':
    #        a,da = ccflibfort.fitccfmax(rv,ccf,sig,ain,lista,funcfitnam)
    #    else:
    #        a,da = ccflibfort.fitccf(rv,ccf,sig,fitrng,ain,lista,funcfitnam)

    if fitrng == 'max':
        a, da = ccflibfort.fitccfmax(rv, ccf, sig, ain, lista, funcfitnam)
    elif fitrng == 'maxabs':
        a, da = ccflibfort.fitccfmaxabs(rv, ccf, sig, ain, lista, funcfitnam)
    else:
        a, da = ccflibfort.fitccf(rv, ccf, sig, fitrng, ain, lista, funcfitnam)

    # Put parameters in dictionary
    fitpar = {
        'amp': a[0], 'cen': a[1], 'wid': a[2], 'shift': a[3],
        'amperr': da[0], 'cenerr': da[1], 'widerr': da[2], 'shifterr': da[3]}
    return fitpar

###############################################################################


# Bisector

def computebisector(x, y, xerr, n=100):
    """
    Compute bisector and its errors.

    Parameters
    ----------
    x, y : 1d arrays
        Data. Must have a Gaussian-like shape.
    xerr : 1d array
        x datapoints errors.
    n : int (default 100)
        Number of points of bisector.

    Returns
    -------
    bx, by : 1d arrays
        Bisector x and y coordinates.
    bxerr : 1d array
        Bisector x datapoints error.
    bx1, bx2 : 1d arrays
        x values at the bisector heights `by` for each side of the line.
    bx1err, bx2err : 1d arrays
        Errors for the data `bx1` and `bx2`.

    """

    # y minimum and maxima (maxima: absolute maxima each side)
    imin = np.nanargmin(y)  # Minimum
    imax1 = np.nanargmax(y[:imin])  # Maximum left part
    imax2 = imin + np.nanargmax(y[imin:])  # Maximum right part
    if imax2 == len(y): imax2p = imax2
    else: imax2p = imax2 + 1  # plus one

    y_smallestmax = np.nanmin([y[imax1], y[imax2]])  # Smallest maximum

    # Bisector y heights
    by = np.linspace(y[imin], y_smallestmax, n)

    # Interpolate bisector y to x for both sides of the y
    #  interp1d(x, y)
    # - Function
    interpolate_x1 = interp1d(y[imax1:imin+1], x[imax1:imin+1], kind='linear')
    interpolate_x2 = interp1d(y[imin:imax2p], x[imin:imax2p], kind='linear')
    # - Bisector x values
    bx1 = interpolate_x1(by)
    bx2 = interpolate_x2(by)

    # Compute bisector
    bx = (bx2 + bx1)/2.

    # -----------------------

    # Bisector error

    # xerr
    # Do not have RVerr for the bisector x datapoints, only for the original x points
    # Solution: Interpolate the error
    # - Function
    interpolate_x1err = interp1d(y[imax1:imin+1], xerr[imax1:imin+1], kind='linear')
    interpolate_x2err = interp1d(y[imin:imax2p], xerr[imin:imax2p], kind='linear')
    # - Bisector x error values
    bx1err = interpolate_x1err(by)
    bx2err = interpolate_x2err(by)

    # Compute bisector error (error propagation)
    bxerr = np.sqrt(bx1err**2 + bx2err**2) / 2.

    return bx, by, bxerr, bx1, bx2, bx1err, bx2err


def computebisector_bis(x, y, ybotmin_percent=10., ybotmax_percent=40., ytopmin_percent=60., ytopmax_percent=90., verb=True):
    """
    Compute bisector inverse slope (BIS).

    Note that the bisector is not interpolated in order to find the points closest to the region limits defined by `ybotmin_percent` etc. So if bisector sampling is low the points won't be correctly selected.

    Parameters
    ----------
    x, y : 1d array like
        Bisector coordinates.
    ybotmin_percent, ybotmax_percent : float
        Bisector bottom region limits in percentage.
    ytopmin_percent, ytopmax_percent : float
        Bisector top region limits in percentage.

    Notes
    -----
    BIS definition from Queloz et al. 2001
    """

    # Check bisector sampling
    s = len(y)
    warn = 'Not good sampling to compute BIS!' if s < 100. else ''
    if verb: print('  {} points in bisector.', warn)

    # Bisector up and down region limits -> absolute value
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    y_delta = y_max - y_min

    ybotmin = y_min + y_delta * ybotmin_percent/100.
    ybotmax = y_min + y_delta * ybotmax_percent/100.
    ytopmin = y_min + y_delta * ytopmin_percent/100.
    ytopmax = y_min + y_delta * ytopmax_percent/100.

    # Bisector up and down region limits indices
    # Note: Approximate regions, depend on bisector sampling
    iybotmin = np.nanargmin(np.abs(y-ybotmin))
    iybotmax = np.nanargmin(np.abs(y-ybotmax))
    iytopmin = np.nanargmin(np.abs(y-ytopmin))
    iytopmax = np.nanargmin(np.abs(y-ytopmax))

    # Compute mean RV in each region
    #### Should it be a weighted mean?
    xmeantop = np.nanmean(x[iytopmin:iytopmax+1])
    xmeanbot = np.nanmean(x[iybotmin:iybotmax+1])

    # Compute bisector inverse slope BIS
    bis = xmeantop - xmeanbot

    return bis, xmeantop, xmeanbot, iybotmin, iybotmax, iytopmin, iytopmax


def computebisector_biserr(x, y, xerr, n=100, bybotmin_percent=10., bybotmax_percent=40., bytopmin_percent=60., bytopmax_percent=90., xrealsampling=None, verb=True, returnall=False):
    """
    Compute bisector, bisector inverse slope (BIS) and their errors.

    Same code as in functions `computebisector` to compute the bisector and in `computebisector_bis` to compute the BIS. Copied here because intermediate products needed to compute the BIS error.

    If do not need the BIS error, use the other functions
    - `computebisector`
    - `computebisector_bis`

    Parameters
    ----------

    x, y : 
    xerr : 
    n : int (default 100)
    bybotmin_percent, bybotmax_percent : float
        Bisector bottom region limits in percentage.
    bytopmin_percent, bytopmax_percent : float
        Bisector top region limits in percentage.
    xrealsampling : float (default None)
        Sampling of the real data. Usually the input data provided (`x` and `y`) is oversampled in order to correctly compute the bisector. So need to provide what would be the real sampling in order to compute the BIS error properly.
    bybotmin_percent ...
    verb
    returnall : bool (default False)
        Output a lot intermediate of products.

    Returns
    -------

    """

    # Bisector
    # --------

    # y minimum and maxima (maxima: absolute maxima each side)
    imin = np.nanargmin(y)  # Minimum
    imax1 = np.nanargmax(y[:imin])  # Maximum left part
    imax2 = imin + np.nanargmax(y[imin:])  # Maximum right part
    if imax2 == len(y): imax2p = imax2
    else: imax2p = imax2 + 1  # plus one

    y_smallestmax = np.nanmin([y[imax1], y[imax2]])  # Smallest maximum

    # Bisector y heights
    by = np.linspace(y[imin], y_smallestmax, n)

    # Interpolate bisector y to x for both sides of the y
    #  interp1d(x, y)
    # - Function
    interpolate_x1 = interp1d(y[imax1:imin+1], x[imax1:imin+1], kind='linear')
    interpolate_x2 = interp1d(y[imin:imax2p], x[imin:imax2p], kind='linear')
    # - Bisector x values
    bx1 = interpolate_x1(by)
    bx2 = interpolate_x2(by)

    # Compute bisector
    bx = (bx2 + bx1)/2.

    # -----------------------

    # Bisector error
    # --------------

    # xerr
    # Do not have RVerr for the bisector x datapoints, only for the original x points
    # Solution: Interpolate the error
    # - Function
    interpolate_x1err = interp1d(y[imax1:imin+1], xerr[imax1:imin+1], kind='linear')
    interpolate_x2err = interp1d(y[imin:imax2p], xerr[imin:imax2p], kind='linear')
    # - Bisector x error values
    bx1err = interpolate_x1err(by)
    bx2err = interpolate_x2err(by)

    # Compute bisector error (error propagation)
    bxerr = np.sqrt(bx1err**2 + bx2err**2) / 2.

    # -----------------------

    # BIS
    # ---

    # Check bisector sampling
    s = len(by)
    warn = 'Not good sampling to compute BIS!' if s < 100. else ''
    if verb: print('  {} points in bisector.'.format(s), warn)

    # Bisector up and down region limits -> absolute value
    by_min = np.nanmin(by)
    by_max = np.nanmax(by)
    by_delta = by_max - by_min

    bybotmin = by_min + by_delta * bybotmin_percent/100.
    bybotmax = by_min + by_delta * bybotmax_percent/100.
    bytopmin = by_min + by_delta * bytopmin_percent/100.
    bytopmax = by_min + by_delta * bytopmax_percent/100.

    # Bisector up and down region limits indices
    # Note: Approximate regions, depend on bisector sampling
    ibybotmin = np.nanargmin(np.abs(by-bybotmin))
    ibybotmax = np.nanargmin(np.abs(by-bybotmax))
    ibytopmin = np.nanargmin(np.abs(by-bytopmin))
    ibytopmax = np.nanargmin(np.abs(by-bytopmax))

    # Compute mean RV in each region
    #### Should it be a weighted mean?
    bxmeantop = np.nanmean(bx[ibytopmin:ibytopmax+1])
    bxmeanbot = np.nanmean(bx[ibybotmin:ibybotmax+1])

    # Compute bisector inverse slope BIS
    bis = bxmeantop - bxmeanbot

    # -----------------------

    # BIS error
    # ---------

    # Compute number of points in top and bottom regions
    #   Use regions x width and x sampling

    # - Original data x sampling
    if xrealsampling is None:
        dx_data = x[1] - x[0]
        if verb:
            print('No real sampling provided for bisector! Need to compute BIS error. Using sampling from input data: {}'.format(dx_data))
    else:
        dx_data = xrealsampling

    # - Top and bottom regions x width (use only one side of the line)
    dx_top = np.abs(interpolate_x1(bytopmax) - interpolate_x1(bytopmin))
    dx_bot = np.abs(interpolate_x1(bybotmax) - interpolate_x1(bybotmin))

    # - Number of points
    ntop = dx_top / dx_data
    nbot = dx_bot / dx_data

    # Mean error top and bottom regions
    bxtopmeanerr = np.nanmean(bxerr[ibybotmin:ibybotmax+1]) / np.sqrt(ntop)
    bxbotmeanerr = np.nanmean(bxerr[ibytopmin:ibytopmax+1]) / np.sqrt(nbot)

    # BIS error
    biserr = np.sqrt(bxtopmeanerr**2 + bxbotmeanerr**2)

    if not returnall:
        return bx, by, bxerr, bis, biserr
    else:
        return bx, by, bxerr, bis, biserr, bx1, bx2, bx1err, bx2err, bxmeantop, bxmeanbot, ibybotmin, ibybotmax, ibytopmin, ibytopmax

###############################################################################


# Utils

def compute_wstd(x, e):
    """
    Compute weighted standard deviation.

    From: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    Returns:
    wstd
    wmean
    """

    masknan = np.isfinite(x) & np.isfinite(e)
    x = x[masknan]
    e = e[masknan]

    # Weights
    w = 1./np.array(e)**2
    # Weighted mean
    wmean = np.average(x, weights=w)
    # Std
    wstd = np.sqrt(np.average((x-wmean)**2, weights=w))
    return wstd, wmean


def determine_fitrng(fitrng, rv, ccf, imin=None, verb=True):
    """
    Determine which points of the CCF use for the fit.

    Parameters
    ----------
    fitrng : str or float
    rv, ccf : 1d array
        CCF datapoints. `rv` only needed if `fitrng` is float. `ccf` only needed for `fitrng` equal to `maxabs`, `maxcl`, float or `all` (actually for `all` only need the array size).
    imin : int or None (default None)
        Position (array index) of the minimum. If None (default) look for the absolute minimum in `ccf`.

    Returns
    -------
    ifit1, ifit2 : int
        Indices of the start and end datapoints of the RV and CCF arrays to use to fit. Select data using array slicing. No need to add `+ 1` to `ifit2` when doing array slicing.
    """
    # From absolute maximum at left side of the minimum to absolute maximum at right side
    if fitrng == 'maxabs':
        ifit1, ifit2, _ = peakutils.min_find_maxabs(ccf, imin=imin)
        # Handle array ends
        if ifit2 < len(ccf): ifit2 += 1

    # From closes maxima at each side of the minimum
    elif fitrng == 'maxcl':
        ifit1, ifit2, _ = peakutils.min_find_maxclosest(ccf, imin=imin)
        # Handle array ends
        if ifit2 < len(ccf): ifit2 = ifit2 + 1

    # All CCF
    elif fitrng == 'all':
        ifit1 = 0
        ifit2 = len(ccf)

    # CCF min +/- rng
    elif pyutils.isfloatnum(fitrng):
        rng = float(fitrng)
        # Minimum
        rvmin = rv[np.nanargmin(ccf)]
        # Range indices
        ifit1 = np.nanargmin(abs(rv - (rvmin - rng)))
        ifit2 = np.nanargmin(abs(rv - (rvmin + rng)))
        # Check if range if within RV array
        if ifit1 == 0 or ifit1 == 1 or ifit2 == len(rv) - 1 or ifit2 == len(rv):
            if verb: print('  Fit range too large, fitting all.')

    return ifit1, ifit2


def selectmask_carmenesgto(filmask, spt, vsini, sptdefault='M3.5', vsinidefault=2.0, verbose=True):
    """
    vsini selection not implemented!!
    Parameters
    ----------
    filmask : str
        Mask-selection file, contaiing the information of the masks to be used. Columns:
            0) object used to make the mask `objmask`
            1) spectral type of `objmask`
            2) `vsini` of `objmask`
            3) path to mask file

        Columns: 0 'objmask': CARMENES id
                 1 'spt': Spectral type of 'objmask'
                 2 'vsini': vsini of 'objmask' [km/s]
                 3 'file': Path to mask file
    spt : str
        Spectral type of the target, e.g. `M3.5`.
    vsini : float
        vsini of the target [km/s].
    sptdefault : str
        Value to use if `spt` is not valid.
    vsinidefault : float
        Value to use if `vsini` is not valid.

    Returns
    -------
    maskinfo : pandas Series
        File and information of the mask selected for 'obj'.
        Labels: 'objmask', 'spt', 'vsini', 'file'.
        E.g. get the file of the mask: `maskinfo['file']`
    """

    # Check valid inputs
    if spt is None:
        stp = sptdefault
    elif len(spt) != 4:
        stp = sptdefault

    if vsini is None:
        vsini = vsinidefault
    elif not np.isfinite(vsini):
        vsini = vsinidefault

    # Read maskdat
    df = pd.read_csv(filmask, sep='\s+', header=0, names=None, usecols=None, skip_blank_lines=True, comment='#')

    # ---- Remove M ----
    # SpT num: M3.5 -> 3.5
    spt = float(spt[1:])
    df['sptnum'] = [float(s[1:]) for s in df['spt']]

    # Available spectral types
    sptavailable = np.sort(np.unique(df['sptnum']))

    # Select masks with the same spectral type as the object
    if spt in sptavailable:
        dfspt = df[df['sptnum'] == spt]
    else:
        sptclosest = sptavailable[np.nanargmin(np.abs(sptavailable - spt))]
        dfspt = df[df['sptnum'] == sptclosest]

    # # Select mask with vsini closest to the object `obj` vsini
    idx = np.nanargmin(np.abs(dfspt['vsini']-vsini))  # In case of multiple occurrences of the minimum values, argmin return the indices corresponding to the first occurrence
    # maskinfo = df.loc[idx]

    maskinfo = dfspt.iloc[0]
    return maskinfo


def showmask_default():
    """List avalilable masks"""
    lisinst = glob.glob(os.path.join(dirhere, 'data/mask/*'))
    print('List available masks for different instruments')
    for inst in lisinst:
        print(os.path.basename(inst))
        lismaskdefault = glob.glob(os.path.join(inst, '*.mas'))
        for mask in lismaskdefault:
            print('  ', os.path.basename(mask))
    return


def listmask_default():
    """Default masks."""
    dictmask = {
        'CARM_VIS': {
            # Mask ID, SpT, vsini [km/s], Mask file
            'J12123+544Sdefault': {'sptmask': 'M0.0V', 'vsinimask': '2.0', 'objmask': 'J12123+544S', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J12123+544Sserval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J11033+359default': {'sptmask': 'M1.5V', 'vsinimask': '2.0', 'objmask': 'J11033+359', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J11033+359serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J19169+051Ndefault': {'sptmask': 'M2.5V', 'vsinimask': '2.0', 'objmask': 'J19169+051N', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J19169+051Nserval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J07274+052default': {'sptmask': 'M3.5V', 'vsinimask': '2.0', 'objmask': 'J07274+052', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J07274+052serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J13229+244default': {'sptmask': 'M4.0V', 'vsinimask': '2.0', 'objmask': 'J13229+244', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J13229+244serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            # 'J13229+244default': {'sptmask': 'M4.0V', 'vsinimask': '2.0', 'objmask': 'J13229+244', 'filmask': pkg_resources.resource_filename('raccoon', 'data/mask/CARM_VIS/J13229+244serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            # 'J13229+244default': {'sptmask': 'M4.0V', 'vsinimask': '2.0', 'objmask': 'J13229+244', 'filmask': pkg_resources.resource_filename('raccoon', 'data/J13229+244serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            # 'J13229+244default': {'sptmask': 'M4.0V', 'vsinimask': '2.0', 'objmask': 'J13229+244', 'filmask': pkg_resources.resource_filename(pkg_resources.Requirement.parse("raccoon"), 'data/mask/CARM_VIS/J13229+244serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J20260+585default': {'sptmask': 'M5.0V', 'vsinimask': '2.0', 'objmask': 'J20260+585', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J20260+585serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J10564+070default': {'sptmask': 'M6.0V', 'vsinimask': '2.9', 'objmask': 'J10564+070', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J10564+070serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J02530+168default': {'sptmask': 'M7.0V', 'vsinimask': '2.0', 'objmask': 'J02530+168', 'filmask': os.path.join(dirhere, 'data/mask/CARM_VIS/', 'J02530+168serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
        },
        'CARM_NIR': {
            # Mask ID, SpT, vsini [km/s], Mask file
            'J12123+544Sdefault': {'sptmask': 'M0.0V', 'vsinimask': '2.0', 'objmask': 'J12123+544S', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J12123+544Sserval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J11033+359default': {'sptmask': 'M1.5V', 'vsinimask': '2.0', 'objmask': 'J11033+359', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J11033+359serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J19169+051Ndefault': {'sptmask': 'M2.5V', 'vsinimask': '2.0', 'objmask': 'J19169+051N', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J19169+051Nserval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J07274+052default': {'sptmask': 'M3.5V', 'vsinimask': '2.0', 'objmask': 'J07274+052', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J07274+052serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J13229+244default': {'sptmask': 'M4.0V', 'vsinimask': '2.0', 'objmask': 'J13229+244', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J13229+244serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J20260+585default': {'sptmask': 'M5.0V', 'vsinimask': '2.0', 'objmask': 'J20260+585', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J20260+585serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J10564+070default': {'sptmask': 'M6.0V', 'vsinimask': '2.9', 'objmask': 'J10564+070', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J10564+070serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
            'J02530+168default': {'sptmask': 'M7.0V', 'vsinimask': '2.0', 'objmask': 'J02530+168', 'filmask': os.path.join(dirhere, 'data/mask/CARM_NIR/', 'J02530+168serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas')},
        },
        'HARPS': {},
        'HARPN': {},
        'EXPRES': {},
        'ESPRESSO': {},
        'ESPRESSO4x2': {},
    }
    return dictmask


def selectmask_default(maskid, inst):
    """Select mask from default available."""
    dictmask = listmask_default()
    mask = dictmask[inst][maskid]
    return mask


def selecttell_default(inst):
    dicttell = {
        'CARM_VIS': os.path.join(dirhere, 'data/tellurics/CARM_VIS/telluric_mask_carm_short.dat'),
        'CARM_NIR': os.path.join(dirhere, 'data/tellurics/CARM_NIR/telluric_mask_nir4.dat'),
    }
    tell = dicttell[inst]
    return tell


def selectfilphoenixf(spt, dirin=os.path.join(dirhere, 'data/phoenix/')):
    """
    - lte02900-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    - lte03000-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    - lte03100-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    - lte03200-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    - lte03300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    - lte03400-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    - lte03900-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    """
    lisfil = np.sort(glob.glob(dirin))
    spt_T = {
        # TODO Check CARM data
        0.0: 3800,
        0.5: 3800,
        1.0: 3600,
        1.5: 3600,
        2.0: 3400,
        2.5: 3400,
        3.0: 3250,
        3.5: 3250,
        4.0: 3100,
        4.5: 3100,
        5.0: 2800,
        5.5: 2800,
        6.0: 2600,
        6.5: 2600,
        7.0: 2500,
        7.5: 2500,
        8.0: 2400,
        8.5: 2400,
        9.0: 2300,
    }

    # TODO

    phoenixfilf = None

    return phoenixfilf

###############################################################################


# Input / Output

def outfits_ccfall(
    rv,
    ccfsum, ccfparsum,  # sum
    ccf, ccfpar,  # orders
    bxsum, bysum,  # sum
    bx, by,  # orders
    headerobs,
    filout,
    ):
    # rv,ccfo,ccfsum,fitparo,ccfpar,fitparsum,ccfparsum,headerspec,filout='ccf.fits'):
    """
    Save CCF data in a FITS file.

    Data saved:
        - Header of the observation -> Header of primary extension
        - RV -> axis
        - CCF sum -> data CCFSUM
        - CCF sum parameters -> header CCFSUM
        - CCF of each order -> data CCFO
        - CCF of each order parameters -> data ccfpar
        - Bisector (x and y) -> BISECTORSUM
            Note: Bisector data inverted wrt CCF data: axis contains flux and data contains RV (because bisector steps constant in flux, not in RV).

    To read data use `infits_ccfall`.
    """

    nord = len(ccf)
    ords = np.arange(0, len(ccf), 1)

    # Observation header
    hdu0 = fits.PrimaryHDU(header=headerobs)

    # -----------------------

    # CCF sum
    hduccfsum = fits.ImageHDU(ccfsum, name='ccfsum')

    hduccfsum.header['CRPIX1'] = (1, 'Reference pixel')
    hduccfsum.header['CRVAL1'] = (rv[0], 'Value of reference pixel [km/s]')
    hduccfsum.header['CDELT1'] = (rv[1]-rv[0], 'Step [km/s]')
    hduccfsum.header['CUNIT1'] = ('km/s', 'Units')

    # -----------------------

    # CCF sum parameters

    # - Save in table?

    # - Save in CCFSUM header

    # None to nan
    ccfparsum = {k: (v if v is not None else np.nan) for k, v in ccfparsum.items()}
    # NaN nor infinite not allowed in FITS header -> str(nan)
    ccfparsum = {k: (v if np.isfinite(v if not isinstance(v, str) else np.nan) else str(v)) for k, v in ccfparsum.items()}

    hduccfsum.header['HIERARCH CCF FIT AMP'] = (ccfparsum['fitamp'], 'Fit')
    hduccfsum.header['HIERARCH CCF FIT CEN'] = (ccfparsum['fitcen'], 'Fit')
    hduccfsum.header['HIERARCH CCF FIT WID'] = (ccfparsum['fitwid'], 'Fit')
    hduccfsum.header['HIERARCH CCF FIT SHIFT'] = (ccfparsum['fitshift'], 'Fit')
    hduccfsum.header['HIERARCH CCF FIT AMPERR'] = (ccfparsum['fitamperr'], 'Fit')
    hduccfsum.header['HIERARCH CCF FIT CENERR'] = (ccfparsum['fitcenerr'], 'Fit')
    hduccfsum.header['HIERARCH CCF FIT WIDERR'] = (ccfparsum['fitwiderr'], 'Fit')
    hduccfsum.header['HIERARCH CCF FIT SHIFTERR'] = (ccfparsum['fitshifterr'], 'Fit')

    hduccfsum.header['HIERARCH CCF RV'] = (ccfparsum['rv'], 'Corrected [km/s]')
    hduccfsum.header['HIERARCH CCF FWHM'] = (ccfparsum['fwhm'], '[km/s]')
    hduccfsum.header['HIERARCH CCF CONTRAST'] = (ccfparsum['contrast'], '[%]')
    hduccfsum.header['HIERARCH CCF BIS'] = (ccfparsum['bis'], '[km/s]')
    hduccfsum.header['HIERARCH CCF RVERR'] = (ccfparsum['rverr'], '[km/s]')
    hduccfsum.header['HIERARCH CCF FWHMERR'] = (ccfparsum['fwhmerr'], '[km/s]')
    hduccfsum.header['HIERARCH CCF CONTRASTERR'] = (ccfparsum['contrasterr'], '[%]')
    hduccfsum.header['HIERARCH CCF BISERR'] = (ccfparsum['biserr'], '[km/s]')

    hduccfsum.header['HIERARCH CCF RVERRABS'] = (ccfparsum['rverrabs'], 'RV err + shift error [km/s]')

    # Comments on the parameters
    hduccfsum.header['COMMENT'] = 'Fit function Gaussian G(x)=shift+amp*e^(-(x-cen)^2/(2*wid^2))'
    hduccfsum.header['COMMENT'] = 'RV=cen'
    hduccfsum.header['COMMENT'] = 'FWHM=2*sqrt(2*ln(2))*wid'
    hduccfsum.header['COMMENT'] = 'CONTRAST=-amp/shift*100'
    hduccfsum.header['COMMENT'] = 'BIS (bisector inverse slope):'
    hduccfsum.header['COMMENT'] = '  Difference between the average bisector'
    hduccfsum.header['COMMENT'] = '  values in the CCF regions from 0.9% to 0.6%'
    hduccfsum.header['COMMENT'] = '  and from 0.4% to 0.1%'
    hduccfsum.header['COMMENT'] = '  (default values, may be different)'

    # RV corrections
    hduccfsum.header['HIERARCH berv'] = (ccfparsum['berv'], '[m/s]')
    hduccfsum.header['HIERARCH drift'] = (ccfparsum['drift'], '[m/s]')
    hduccfsum.header['HIERARCH sa'] = (ccfparsum['sa'], '[m/s]')
    hduccfsum.header['HIERARCH berverr'] = (ccfparsum['berverr'], '[m/s]')
    hduccfsum.header['HIERARCH drifterr'] = (ccfparsum['drifterr'], '[m/s]')
    hduccfsum.header['HIERARCH saerr'] = (ccfparsum['saerr'], '[m/s]')
    hduccfsum.header['HIERARCH shift'] = (ccfparsum['shift'], '-BERV+drift+sa+otherdrift [m/s]')
    hduccfsum.header['HIERARCH shifterr'] = (ccfparsum['shifterr'], '[m/s]')
    hduccfsum.header['HIERARCH otherdrift'] = (ccfparsum['otherdrift'], '[m/s]')
    hduccfsum.header['HIERARCH otherdrifterr'] = (ccfparsum['otherdrifterr'], '[m/s]')

    # Other observation data
    hduccfsum.header['HIERARCH BJD'] = (ccfparsum['bjd'], 'BJD')
    hduccfsum.header['HIERARCH REFERENCE ORDER'] = (ccfparsum['oref'], 'oref')
    hduccfsum.header['HIERARCH SNR REFERENCE ORDER'] = (ccfparsum['snroref'], 'snroref')
    hduccfsum.header['HIERARCH READOUT NOISE'] = (ccfparsum['ron'], 'ron')
    hduccfsum.header['HIERARCH EXPTIME'] = (ccfparsum['exptime'], 'exptime')

    # Mask and mask number of lines
    hduccfsum.header['MASKFIL'] = (ccfparsum['filmask'], 'Mask file')
    hduccfsum.header['MASKFILN'] = (ccfparsum['filmaskname'], 'Mask file name')
    hduccfsum.header['HIERARCH MASK TARGET'] = (ccfparsum['objmask'], 'Mask target')
    hduccfsum.header['HIERARCH MASK TARGET SPT'] = (ccfparsum['sptmask'], 'Mask target SpT')
    hduccfsum.header['HIERARCH MASK TARGET VSINI'] = (ccfparsum['vsinimask'], 'Mask target vsini')
    for o in ords:
        hduccfsum.header['HIERARCH MASK NLINO{}'.format(o)] = (ccfparsum['nlino{}'.format(o)], 'Num mask lines order {}'.format(o))
    hduccfsum.header['HIERARCH MASK NLIN TOTAL'] = (ccfparsum['nlint'], 'Total num of mask lines (taking into account BERV, mask shift, order overlap)')
    hduccfsum.header['HIERARCH MASK NLIN ORIGINAL'] = (ccfparsum['nlinoriginal'], 'Num of mask lines original mask')

    # -----------------------

    # CCF orders
    hduccfo = fits.ImageHDU(ccf, name='ccfo')

    # Cannot have nan in header. If all nan -> 0
    if (~np.isfinite(rv)).all(): rv = np.zeros_like(rv)

    hduccfo.header['CRPIX1'] = (1, 'Reference pixel')
    hduccfo.header['CRVAL1'] = (rv[0], 'Value of reference pixel [km/s]')
    hduccfo.header['CDELT1'] = (rv[1]-rv[0], 'Step [km/s]')
    hduccfo.header['CUNIT1'] = ('km/s', 'Units')

    # -----------------------

    # CCF orders parameters
    hduccfpar = fits.TableHDU(ccfpar.to_records(), name='ccfparo')
    hduccfpar.header['COMMENT'] = 'fitamp, fitamperr, fitcen, fitcenerr, fitwid, fitwiderr, fitshift, fitshifterr'
    hduccfpar.header['COMMENT'] = 'RV, RVERR, FWHM, FWHMERR [km/s]'
    hduccfpar.header['COMMENT'] = 'CONTRAST, CONTRASTERR [%]'
    hduccfpar.header['COMMENT'] = 'BIS, BISERR [km/s]'

    # -----------------------

    # Bisector
    #  Bisector steps constant in CCF, not in RV
    hdubisectorsum = fits.ImageHDU(bxsum, name='bisectorsum')

    # Cannot have nan in header. If all nan -> 0
    if (~np.isfinite(bysum)).all(): bysum = np.zeros_like(bysum)

    hdubisectorsum.header['CRPIX1'] = (1, 'Reference pixel')
    hdubisectorsum.header['CRVAL1'] = (bysum[0], 'Value of reference pixel [flux]')
    hdubisectorsum.header['CDELT1'] = (bysum[1]-bysum[0], 'Step [flux]')
    hdubisectorsum.header['CUNIT1'] = ('flux', 'Units')

    # -----------------------

    # Bisector orders
    hdubisectorrvo = fits.ImageHDU(bx, name='bisectorrvo')
    hdubisectorccfo = fits.ImageHDU(by, name='bisectorccfo')

    # -----------------------

    # Save
    hdulist = fits.HDUList([hdu0, hduccfsum, hduccfo, hduccfpar, hdubisectorsum, hdubisectorrvo, hdubisectorccfo])
    # If problems with original header -> Remove it
    try:
        hdulist.writeto(filout, overwrite=True, output_verify='silentfix+warn')
    except ValueError:
        hdu0 = fits.PrimaryHDU()  # replace original header
        hdulist = fits.HDUList([hdu0, hduccfsum, hduccfo, hduccfpar, hdubisectorsum, hdubisectorrvo, hdubisectorccfo])
        hdulist.writeto(filout, overwrite=True, output_verify='silentfix+warn')
    return


def infits_ccfall(filin):
    """Read CCF data from files created with `outfits_ccfall`.

    Returns
    -------
    rv : 1d-array
        RV grid of the CCFs.
    ccfsum : 1d-array
        CCF of all orders coadded (same length as `rv`).
    ccfparsum : dict
        Parameters of the coadded CCF.
    ccf : nd-array
        CCFs of each order. Shape: (number of orders, RV grid length).
    ccfpar : pandas DataFrame
        Parameters of the CCF of each order.
    bxsum : 1d-array
        Bisector of the coadded CCF (x-axis). Length: 100 points (unless changed when computing the bisector).
    bysum : 1d-array
        Bisector of the coadded CCF (y-axis). Length: 100 points (unless changed when computing the bisector).
    bx : nd-array
        Bisectors of the CCF of each order (x-axis). Shape: (number of orders, 100 (unless changed when computing the bisector)).
    by : nd-array
        Bisectors of the CCF of each order (y-axis). Shape: (number of orders, 100 (unless changed when computing the bisector)).
    headerobs : astropy Header
        Header of the original reduced spectrum file.

    FITS info example
    -----------------

    >>> hdulist.info()
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     340   ()
      1  CCFSUM        1 ImageHDU        36   (129,)   float64
      2  CCFO          1 ImageHDU        12   (129, 61)   float64
      3  CCFPAR        1 TableHDU        62   61R x 16C   [I21, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17, D25.17]
      4  BISECTORSUM    1 ImageHDU        11   (100,)   float64
      5  BISECTORRVO    1 ImageHDU         8   (100, 61)   float64
      6  BISECTORCCFO    1 ImageHDU         8   (100, 61)   float64
    """
    with fits.open(filin) as hdulist:

        nord = len(hdulist['ccfo'].data)
        ords = np.arange(0, nord, 1)

        # Observation header
        headerobs = hdulist[0].header

        # RV
        h = hdulist['ccfsum'].header
        n = h['NAXIS1']
        stp = h['CDELT1']
        ini_pix = h['CRPIX1']
        ini_val = h['CRVAL1']
        # fin_val = ini_val + (n - (ini_pix - 1)) * stp
        fin_val = ini_val + n * stp
        rv = np.arange(ini_val, fin_val, stp)

        # Issue with decimals: rv array can end up having an extra point
        # E.g. J09439+269: From FITS header, initial RV value is `ini_val = 19.1788947345789` but from the original RV array, initial RV value is `rv[0]=19.178894734578897`.
        if len(rv) == n+1: rv = rv[:-1]
        elif len(rv) != n: print('rv array length wrong')

        # CCF sum
        ccfsum = hdulist['ccfsum'].data

        # CCF sum parameters
        ccfparsum = {}
        h = hdulist['ccfsum'].header

        ccfparsum['fitamp'] = h['HIERARCH CCF FIT AMP']
        ccfparsum['fitcen'] = h['HIERARCH CCF FIT CEN']
        ccfparsum['fitwid'] = h['HIERARCH CCF FIT WID']
        ccfparsum['fitshift'] = h['HIERARCH CCF FIT SHIFT']
        ccfparsum['fitamperr'] = h['HIERARCH CCF FIT AMPERR']
        ccfparsum['fitcenerr'] = h['HIERARCH CCF FIT CENERR']
        ccfparsum['fitwiderr'] = h['HIERARCH CCF FIT WIDERR']
        ccfparsum['fitshifterr'] = h['HIERARCH CCF FIT SHIFTERR']

        ccfparsum['rv'] = h['HIERARCH CCF RV']
        ccfparsum['fwhm'] = h['HIERARCH CCF FWHM']
        ccfparsum['contrast'] = h['HIERARCH CCF CONTRAST']
        ccfparsum['bis'] = h['HIERARCH CCF BIS']
        ccfparsum['rverr'] = h['HIERARCH CCF RVERR']
        ccfparsum['fwhmerr'] = h['HIERARCH CCF FWHMERR']
        ccfparsum['contrasterr'] = h['HIERARCH CCF CONTRASTERR']
        ccfparsum['biserr'] = h['HIERARCH CCF BISERR']

        ccfparsum['rverrabs'] = h['HIERARCH CCF RVERRABS']
        # RV corrections
        ccfparsum['berv'] = h['HIERARCH berv']
        ccfparsum['drift'] = h['HIERARCH drift']
        ccfparsum['sa'] = h['HIERARCH sa']
        ccfparsum['berverr'] = h['HIERARCH berverr']
        ccfparsum['drifterr'] = h['HIERARCH drifterr']
        ccfparsum['saerr'] = h['HIERARCH saerr']
        ccfparsum['shift'] = h['HIERARCH shift']
        ccfparsum['shifterr'] = h['HIERARCH shifterr']
        ccfparsum['otherdrift'] = h['HIERARCH otherdrift']
        ccfparsum['otherdrifterr'] = h['HIERARCH otherdrifterr']

        # Other observation data
        ccfparsum['bjd'] = h['HIERARCH BJD']
        ccfparsum['oref'] = h['HIERARCH REFERENCE ORDER']
        ccfparsum['snroref'] = h['HIERARCH SNR REFERENCE ORDER']
        ccfparsum['ron'] = h['HIERARCH READOUT NOISE']
        ccfparsum['exptime'] = h['HIERARCH EXPTIME']

        # Mask and mask number of lines
        ccfparsum['filmask'] = h['MASKFILN']
        ccfparsum['filmaskname'] = h['MASKFILN']
        ccfparsum['objmask'] = h['HIERARCH MASK TARGET']
        ccfparsum['sptmask'] = h['HIERARCH MASK TARGET SPT']
        ccfparsum['vsinimask'] = h['HIERARCH MASK TARGET VSINI']
        for o in ords:
            ccfparsum['nlino{}'.format(o)] = h['HIERARCH MASK NLINO{}'.format(o)]
        ccfparsum['nlint'] = h['HIERARCH MASK NLIN TOTAL']
        ccfparsum['nlinoriginal'] = h['HIERARCH MASK NLIN ORIGINAL']

        for k, v in ccfparsum.items():
            if v == 'nan' or v == 'inf' or v == '-inf':
                ccfparsum[k] = np.nan

        # CCF orders
        ccf = hdulist['ccfo'].data

        # CCF orders parameters
        ccfpar = pd.DataFrame.from_records(hdulist['ccfparo'].data, index='orders')
        ccfpar.index.set_names('orders', inplace=True)

        # Bisector CCF sum
        bxsum = hdulist['bisectorsum'].data
        h = hdulist['bisectorsum'].header
        n = h['NAXIS1']
        stp = h['CDELT1']
        ini_pix = h['CRPIX1']
        ini_val = h['CRVAL1']
        fin_val = ini_val + n * stp
        try: bysum = np.arange(ini_val, fin_val, stp)
        except: bysum = np.ones_like(bxsum) * np.nan

        # Bisector orders
        bx = hdulist['bisectorrvo'].data
        by = hdulist['bisectorccfo'].data

    return rv, ccfsum, ccfparsum, ccf, ccfpar, bxsum, bysum, bx, by, headerobs


def outdat_ccf(filout, rv, ccf):
    """Save single CCF data in text file. Columns: 0) RV, 1) CCF.
    """
    data = np.vstack([rv, ccf])
    np.savetxt(filout, data.T, fmt='%.8f')
    return


def outdat_ccfparTS(filout, data, cols=['bjd', 'rv', 'fwhm', 'contrast', 'bis', 'rverr', 'fwhmerr', 'contrasterr', 'biserr'], sep=' ', na_rep=np.nan, header=False, index=False, float_format='%0.8f'):
    """Save CCF parameters TS in text file.

    Parameters
    ----------
    data : pandas DataFrame
    cols : list
        Names of the `data` columns to be saved to file. The columns must exists in the DataFrame.
        Default: ['bjd', 'rv', 'fwhm', 'contrast', 'bis', 'rverr', 'fwhmerr', 'contrasterr', 'biserr']
    """
    data.to_csv(filout, columns=cols, sep=sep, na_rep=na_rep, header=header, index=index, float_format=float_format)
    return

###############################################################################


# Plots

pmstr = r"$\pm$"
# pmstr = u'\u00B1'.encode('utf-8')


def plot_ccf(rv, ccf, ccfpar=None, title='', **kwargs):
    fig, ax = plt.subplots()

    # CCF
    ax.plot(rv, ccf, 'ko', **kwargs)

    # Info
    if ccfpar is not None:
        infoobs = 'BJD {:.2f}\nSNR{} {:.0f}'.format(ccfpar['bjd'], ccfpar['oref'], ccfpar['snroref'])
        infomask = 'nlin {}, {}'.format(ccfpar['nlinoriginal'], ccfpar['nlint'])
        infoccfpar = 'RV {:.2f}{}{:.2f} ({:.2f}) m/s\nFWHM {:.2f}{}{:.2f} km/s\nContrast {:.2f}{}{:.2f} %\nBIS {:.2f}{}{:.2f} m/s'.format(ccfpar['rv']*1.e3, pmstr, ccfpar['rverr']*1.e3, ccfpar['rverrabs']*1.e3, ccfpar['fwhm'], pmstr, ccfpar['fwhmerr'], ccfpar['contrast'], pmstr, ccfpar['contrasterr'], ccfpar['bis']*1.e3, pmstr, ccfpar['biserr']*1.e3)
        infoccffit = 'amp {:.2f}{}{:.2f}\ncen {:.2f}{}{:.2f}\nwid {:.2f}{}{:.2f}\nshift {:.2f}{}{:.2f}'.format(ccfpar['fitamp']*1.e3, pmstr, ccfpar['fitamperr']*1.e3, ccfpar['fitcen'], pmstr, ccfpar['fitcenerr'], ccfpar['fitwid'], pmstr, ccfpar['fitwiderr'], ccfpar['fitshift']*1.e3, pmstr, ccfpar['fitshifterr']*1.e3)

        # - infobs + infomask + infoccfpar lower left
        bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
        x, y = 0.03, 0.03
        ax.text(x, y, infoobs + '\n' + infomask + '\n\n' + infoccfpar, ha='left', va='bottom', transform=ax.transAxes, fontsize='xx-small', color='k', bbox=bbox_props)

    ax.set_xlabel('RV [km/s]')
    # ax.set_ylabel('CCF')
    plotutils.Labeloffset(ax, label="CCF", axis="y")
    ax.set_title(title)
    ax.minorticks_on()
    return fig, ax


def plot_ccf_fit(rv, ccf, ccfpar, title='', fitpar='', **kwargs):
    fig, ax = plt.subplots()

    # CCF
    ax.plot(rv, ccf, 'ko', **kwargs)

    # Fit
    fit = peakutils.gaussian(rv, amp=ccfpar['fitamp'], cen=ccfpar['fitcen'], wid=ccfpar['fitwid'], shift=ccfpar['fitshift'])
    ax.plot(rv, fit, 'C1--')

    # Info
    infoobs = 'BJD {:.2f}\nSNR{} {:.0f}'.format(ccfpar['bjd'], ccfpar['oref'], ccfpar['snroref'])
    infomask = 'nlin {}, {}'.format(ccfpar['nlinoriginal'], ccfpar['nlint'])
    infoccfpar = 'RV {:.2f}{}{:.2f} ({:.2f}) m/s\nFWHM {:.2f}{}{:.2f} km/s\nContrast {:.2f}{}{:.2f} %\nBIS {:.2f}{}{:.2f} m/s'.format(ccfpar['rv']*1.e3, pmstr, ccfpar['rverr']*1.e3, ccfpar['rverrabs']*1.e3, ccfpar['fwhm'], pmstr, ccfpar['fwhmerr'], ccfpar['contrast'], pmstr, ccfpar['contrasterr'], ccfpar['bis']*1.e3, pmstr, ccfpar['biserr']*1.e3)
    infoccffit = 'amp {:.2e}{}{:.2e}\ncen {:.2f}{}{:.2f}\nwid {:.2f}{}{:.2f}\nshift {:.2e}{}{:.2e}'.format(ccfpar['fitamp']*1.e3, pmstr, ccfpar['fitamperr']*1.e3, ccfpar['fitcen'], pmstr, ccfpar['fitcenerr'], ccfpar['fitwid'], pmstr, ccfpar['fitwiderr'], ccfpar['fitshift']*1.e3, pmstr, ccfpar['fitshifterr']*1.e3)

    # - infobs + infomask + infoccfpar lower left
    bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
    x, y = 0.03, 0.03
    ax.text(x, y, infoobs + '\n' + infomask + '\n\n' + infoccfpar, ha='left', va='bottom', transform=ax.transAxes, fontsize='xx-small', color='k', bbox=bbox_props) 

    # - infoccffit lower right
    bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
    x, y = 1.-0.03, 0.03
    ax.text(x, y, infoccffit, ha='right', va='bottom', transform=ax.transAxes, fontsize='xx-small', color='k', bbox=bbox_props) 

    ax.set_xlabel('RV [km/s]')
    plotutils.Labeloffset(ax, label="CCF", axis="y")
    #ax.set_ylabel('CCF')
    # Add fit params text
    # ...
    ax.set_title(title)
    ax.minorticks_on()
    return fig, ax


def plot_ccf_fit_line(rv, ccf, ccfpar, title='', fitpar='', **kwargs):
    fig, ax = plt.subplots()

    # CCF
    ax.plot(rv, ccf, 'k-', **kwargs)

    # Fit
    fit = peakutils.gaussian(rv, amp=ccfpar['fitamp'], cen=ccfpar['fitcen'], wid=ccfpar['fitwid'], shift=ccfpar['fitshift'])
    ax.plot(rv, fit, 'C1--')

    # Info
    infoobs = 'BJD {:.2f}\nSNR{} {:.0f}'.format(ccfpar['bjd'], ccfpar['oref'], ccfpar['snroref'])
    infomask = 'nlin {}, {}'.format(ccfpar['nlinoriginal'], ccfpar['nlint'])
    infoccfpar = 'RV {:.2f}{}{:.2f} ({:.2f}) m/s\nFWHM {:.2f}{}{:.2f} km/s\nContrast {:.2f}{}{:.2f} %\nBIS {:.2f}{}{:.2f} m/s'.format(ccfpar['rv']*1.e3, pmstr, ccfpar['rverr']*1.e3, ccfpar['rverrabs']*1.e3, ccfpar['fwhm'], pmstr, ccfpar['fwhmerr'], ccfpar['contrast'], pmstr, ccfpar['contrasterr'], ccfpar['bis']*1.e3, pmstr, ccfpar['biserr']*1.e3)
    infoccffit = 'amp {:.2e}{}{:.2e}\ncen {:.2f}{}{:.2f}\nwid {:.2f}{}{:.2f}\nshift {:.2e}{}{:.2e}'.format(ccfpar['fitamp']*1.e3, pmstr, ccfpar['fitamperr']*1.e3, ccfpar['fitcen'], pmstr, ccfpar['fitcenerr'], ccfpar['fitwid'], pmstr, ccfpar['fitwiderr'], ccfpar['fitshift']*1.e3, pmstr, ccfpar['fitshifterr']*1.e3)

    # - infobs + infomask + infoccfpar lower left
    bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
    x, y = 0.03, 0.03
    ax.text(x, y, infoobs + '\n' + infomask + '\n\n' + infoccfpar, ha='left', va='bottom', transform=ax.transAxes, fontsize='xx-small', color='k', bbox=bbox_props) 

    # - infoccffit lower right
    bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
    x, y = 1.-0.03, 0.03
    ax.text(x, y, infoccffit, ha='right', va='bottom', transform=ax.transAxes, fontsize='xx-small', color='k', bbox=bbox_props) 

    ax.set_xlabel('RV [km/s]')
    plotutils.Labeloffset(ax, label="CCF", axis="y")
    #ax.set_ylabel('CCF')
    # Add fit params text
    # ...
    ax.set_title(title)
    ax.minorticks_on()
    return fig, ax


def plot_ccf_fit_diff(rv, ccf, ccfpar, title='', fitpar='', diffzoom=True, parvelunit='kms', **kwargs):
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]})

    # CCF
    ax[0].plot(rv, ccf, 'ko', **kwargs)

    # Fit
    fit = peakutils.gaussian(rv, amp=ccfpar['fitamp'], cen=ccfpar['fitcen'], wid=ccfpar['fitwid'], shift=ccfpar['fitshift'])
    ax[0].plot(rv, fit, 'C1--')

    # Data units
    if parvelunit == 'kms': facv = 1.
    elif parvelunit == 'ms': facv = 1.e3

    # Info
    try: infoobs = 'BJD {:.2f}\nSNR{} {:.0f}'.format(ccfpar['bjd'], ccfpar['oref'], ccfpar['snroref'])
    except: infoobs = ''

    try: infomask = 'nlin {}, {}'.format(ccfpar['nlinoriginal'], ccfpar['nlint'])
    except: infomask = ''

    try: infoccfpar = 'RV {:.2f}{}{:.2f} ({:.2f}) m/s\nFWHM {:.2f}{}{:.2f} km/s\nContrast {:.2f}{}{:.2f} %\nBIS {:.2f}{}{:.2f} m/s'.format(ccfpar['rv']*1.e3, pmstr, ccfpar['rverr']*1.e3, ccfpar['rverrabs']*1.e3, ccfpar['fwhm'], pmstr, ccfpar['fwhmerr'], ccfpar['contrast'], pmstr, ccfpar['contrasterr'], ccfpar['bis']*1.e3, pmstr, ccfpar['biserr']*1.e3)
    except: infoccfpar = ''

    try: infoccffit = 'amp {:.2e}{}{:.2e}\ncen {:.2f}{}{:.2f}\nwid {:.2f}{}{:.2f}\nshift {:.2e}{}{:.2e}'.format(ccfpar['fitamp']*1.e3, pmstr, ccfpar['fitamperr']*1.e3, ccfpar['fitcen']*facv, pmstr, ccfpar['fitcenerr']*facv, ccfpar['fitwid'], pmstr, ccfpar['fitwiderr'], ccfpar['fitshift']*1.e3, pmstr, ccfpar['fitshifterr']*1.e3)
    except: infoccffit = ''

    # - infobs + infomask + infoccfpar lower left
    bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
    x, y = 0.03, 0.03
    ax[0].text(x, y, infoobs + '\n' + infomask + '\n\n' + infoccfpar, ha='left', va='bottom', transform=ax[0].transAxes, fontsize='xx-small', color='k', bbox=bbox_props)

    # - infoccffit lower right
    bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
    x, y = 1.-0.03, 0.03
    ax[0].text(x, y, infoccffit, ha='right', va='bottom', transform=ax[0].transAxes, fontsize='xx-small', color='k', bbox=bbox_props)

    # Difference
    diff = ccf - fit
    if diffzoom:
        icen = len(rv) / 2.
        di = len(rv) / 8.
        mask = [True if (i >= icen - di) and (i <= icen + di) else False for i in range(len(rv))]
        ax[1].plot(rv[mask], diff[mask], 'ko')
    else:
        ax[1].plot(rv, diff, 'ko')

    ax[-1].set_xlabel('RV [km/s]')
    # plotutils.Labeloffset(ax[0], label="CCF", axis="y")
    ax[0].set_ylabel('CCF')
    # Add fit params text
    # ...
    ax[0].set_title(title)
    for a in ax:
        a.minorticks_on()
    return fig, ax[0], ax[1]


def plot_ccfo_lines(rv, ccfo, lisord=None, printord=True, multiline=True, cb=True, cmap=cc.cm.rainbow4, lw=2, alpha=0.9, xlabel='RV [km/s]', ylabel='Order CCF', cblabel='Order', ax=None, **kwargs):
    # cmap = cc.cm.CET_I1
    # cmap = 'Spectral'
    """Plot order CCFs as given, no normalisation or offset.
    
    Parameters
    ----------
    rv : 1D array-like
        RV grid of the CCF
    ccfo : 2D array-like
        CCFs of each order
    lisord : 1D array-like
        Order number of each CCF in `ccfo`. If None, order numbers are 0 to `len(ccfo)`. Only shown if `printord` is `True` (default).
    multiline : bool, default True
        If True plot lines following a specific color map (default). If not, color-code following default color-cycle.
    """
    # Make sure we have an axis where to plot the fit
    if not isinstance(ax, mpl.axes._subplots.Axes): ax = plt.gca()
    # Order number
    if lisord is None: lisord = np.arange(0, len(ccfo), 1)
    # Plot
    if multiline == False:
        # Plot lines following default color cycle
        for o, ccf in zip(lisord, ccfo):
            # CCF
            ax.plot(rv, ccf, cmap=cmap, lw=lw, alpha=alpha, **kwargs)
            # Order number TODO
            # ax.text()
    else:
        # Plot lines following a specific map
        # CCF
        xs = np.array([rv for i in ccfo])
        ys = np.array(ccfo)
        c = np.array(lisord)
        lc = plotutils.multiline(xs, ys, c, cmap=cmap, lw=lw, alpha=alpha, ax=ax, **kwargs)
        # Colorbar
        if cb:
            cbar = plt.colorbar(lc, ax=ax, aspect=15, pad=0.02, label=cblabel)
            cbar.minorticks_on()
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_ccfo_map(rv, ccfo, lisord=None, cb=True, cmap=cmocean.cm.deep_r, extent='data', vmin=None, vmax=None, extend='neither', xlabel='RV [km/s]', ylabel='Order', cblabel='CCF', ax=None):
    """Plot map of order CCFs as given, no normalisation or offset.
    
    Parameters
    ----------
    rv : 1D array-like
        RV grid of the CCF
    ccfo : 2D array-like
        CCFs of each order
    lisord : 1D array-like
        Order number of each CCF in `ccfo`. If None, order numbers are 0 to `len(ccfo)`. Only shown if `printord` is `True` (default).
    """
    # Make sure we have an axis where to plot the fit
    if not isinstance(ax, mpl.axes._subplots.Axes): ax = plt.gca()

    # Orde rnumber
    if lisord is None: lisord = np.arange(0, len(ccfo), 1)
    # Plot
    if extent == 'data': extent = [rv[0], rv[-1], lisord[0], lisord[-1]]
    nocb = not cb
    ax = plotutils.plot_map(ccfo, cblabel, interpolation='none', origin='lower', extent=extent, vmin=vmin, vmax=vmax, extend=extend, cmap=cmap, axcb=None, nocb=nocb, aspect=15, pad=0.02, fraction=0.15, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_ccfo_lines_map(rv, ccfo, ccfsum, ccferrsum, lisord=None, 
    ylabelsum = 'Coadd. CCF',
    multiline=True, cbline=True, cmapline=cc.cm.rainbow4, lw=2, alpha=0.9, ylabelline='Order CCF', cblabelline='Order',
    cbmap=True, cmapmap=cmocean.cm.deep_r, extent='data', vmin=None, vmax=None, extend='neither', ylabelmap='Order', cblabelmap='CCF',
    xlabel='RV [km/s]', title=''
    ):
    """3-panel plot with CCFsum, CCFo lines, CCFo map, one below each other.
    """
    fig, ax = plt.subplots(3,1, figsize=(8,7), sharex=True, gridspec_kw={'height_ratios': [1,2.5,2.5]})
    axsum = ax[0]
    axo = ax[1]
    axmap = ax[2]
    # CCF sum
    axsum.errorbar(rv, ccfsum, yerr=ccferrsum, fmt='k.')
    axsum.set_ylabel(ylabelsum)
    # CCFo
    axo = plot_ccfo_lines(rv, ccfo, cmap=cmapline, lw=lw, alpha=alpha, xlabel=None, ylabel=ylabelline, ax=axo)
    # CCFo map
    axmap = plot_ccfo_map(rv, ccfo, cmap=cmapmap, extent=extent, vmin=vmin, vmax=vmax, extend=extend, xlabel=xlabel, ylabel=ylabelmap, ax=axmap)
    # 
    ax[0].set_title(title)
    for a in ax.flatten():
        a.minorticks_on()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)  # wspace=0.08, hspace=0.08
    # Adjust ccfsum panel size
    boxo = axo.get_position()
    boxsum = axsum.get_position()
    # Set based on width
    axsum.set_position([boxsum.x0, boxsum.y0, boxo.width, boxsum.height])  # left, bottom, width, height
    # # Or set with Bbox and coordinates
    # boxsum_new = mpl.transforms.Bbox([[boxsum.x0, boxsum.y0], [boxo.x1, boxsum.y1]])
    # axsum.set_position(boxsum_new)
    # 
    return fig, axsum, axo, axmap



def plot_lisccf_lines_map(rv, lisccf, lisccferr, time, maskgray=None, maskmap=None, labelrv='RV [km/s]', labelccf='CCF',
    cmapline=cc.cm.rainbow4, lw=2, alpha=0.9, cbline=True, labeltime='Time',
    interpolation='none', origin='lower', extent='data', vmin=None, vmax=None, extend='neither', cbmap=True, cmapmap=cmocean.cm.deep_r,
    title='',):
    """2-panel plot with list of CCF (coadded CCFs from different observations) lines color-coded as a function of time, and CCF map with time on y-axis and color-ocoded on CCF value, one below each other.

    TODO: add errorbars `lisccferr`

    maskgray : 1D array-like
        Plot set of CCFs in gray (e.g. out-of-transit, or low S/N)
    """
    if maskgray is None: maskgray = np.zeros_like(time, dtype=bool)
    fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios': [3,1.2]})
    axline = ax[0]
    axmap = ax[1]
    # CCF lines
    xs = np.array([rv for i in lisccf[~maskgray]])
    ys = np.array(lisccf[~maskgray])
    c = np.array(time[~maskgray])
    lc = plotutils.multiline(xs, ys, c, cmap=cmapline, lw=lw, alpha=alpha, ax=axline)
    # Colorbar
    if cbline:
        cbar = plt.colorbar(lc, ax=axline, aspect=15, pad=0.02, label=labeltime)
        cbar.minorticks_on()
    axline.set_ylabel(labelccf)

    # CCF line gray
    for ccf in lisccf[maskgray]:
        axline.plot(rv, ccf, '0.5', lw=1, alpha=alpha, zorder=0)

    # CCF map
    if maskmap is None: maskmap = np.ones_like(time, dtype=bool)
    nocb = not cbmap
    # 
    # If set the aspect of the small panel to be the same of the big on, the cb width is narrower and it looks weird. Hence need to change it as follows:
    #   Width of cb of small panel to be the same of bigger panel:
    #       aspect_cb_bigpanel = 15
    #       bigpanel_height = 3
    #       smallpanel_height = 1.2
    #       aspect_cb_smallpanel = aspect_cb_bigpanel * smallpanel_height / bigpanel_height = 6
    # 
    # Check if observations are non-consecutive
    consecutiveobs = True
    time_min = np.diff(time[maskmap]).min()
    time_min_range = 2
    for dt in np.diff(time[maskmap]):
        if dt > time_min*time_min_range:
            consecutiveobs = False
            break
    # If obs are non-consecutive, imshow will show the wrong "grid" because it assumes the values are equally spaced. Need to add "fake" nan-filled CCFs in between non-consecutive obs.
    # Add a nan-filled CCF at the beginning and end because if not the actual ones might be hidden by the borders
    # For non-uniform grid spacing, use plt.pcolor(rv, time, lisccf) or pcolormesh. But have everything coded with imshow, so it's easier to tweak the data.
    if consecutiveobs is False:
        # Add nan-filled CCF between non-consecutive observations
        # Define consecutive by twice the minimum difference in time
        time_min = np.diff(time[maskmap]).min() * 2
        time_new = []
        lisccf_new = []
        maskmap_new = []
        nanccf = np.ones_like(lisccf[0])*np.nan
        # Fake obs beginning
        time_new.append(time[maskmap][0]-time_min)
        lisccf_new.append(nanccf)
        maskmap_new.append(True)
        for obs_i in range(len(time[maskmap])):
            time_new.append(time[maskmap][obs_i])
            lisccf_new.append(lisccf[obs_i])
            maskmap_new.append(maskmap[obs_i])
            # Last obs
            if obs_i == len(time[maskmap])-1:
                continue
            # Time diff with following observation
            dt = time[maskmap][obs_i+1] - time[maskmap][obs_i]
            # If difference larger than minimum time, assume non-consecutive obs, and fill with nan
            if dt > time_min:
                # How many new nan-filled obs are needed
                nobs_new = int(dt/time_min)
                for newobs in range(nobs_new):
                    time_new.append(time[maskmap][obs_i]*(newobs+2))
                    lisccf_new.append(nanccf)
                    maskmap_new.append(True)
        # Fake obs end
        time_new.append(time[maskmap][-1]+time_min)
        lisccf_new.append(nanccf)
        maskmap_new.append(True)
        # Arrays
        time = np.array(time_new)
        lisccf = np.array(lisccf_new)
        maskmap = np.array(maskmap_new)
    # 
    if extent == 'data': extent = [rv[0], rv[-1], time[maskmap][0], time[maskmap][-1]]
    axmap = plotutils.plot_map(lisccf[maskmap], labelccf, interpolation=interpolation, origin=origin, extent=extent, vmin=vmin, vmax=vmax, extend=extend, cmap=cmapmap, axcb=None, nocb=nocb, aspect=6, pad=0.02, fraction=0.15, ax=axmap)

    axmap.set_ylabel(labeltime)
    ax[-1].set_xlabel(labelrv)

    ax[0].set_title(title)
    for a in ax.flatten():
        a.minorticks_on()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)  # wspace=0.08, hspace=0.08

    # Adjust ccfsum panel size (if 1st panel has no colorbar)
    if cbline == False:
        boxo = axo.get_position()
        boxsum = axsum.get_position()
        # Set based on width
        axsum.set_position([boxsum.x0, boxsum.y0, boxo.width, boxsum.height])  # left, bottom, width, height
        # # Or set with Bbox and coordinates
        # boxsum_new = mpl.transforms.Bbox([[boxsum.x0, boxsum.y0], [boxo.x1, boxsum.y1]])
        # axsum.set_position(boxsum_new)
    return fig, axline, axmap


def plot_ccf_bisector():
    pass
    return


def printstats(stats, ax, data, err=None, x=0.97, y=0.93, ha='right', va='top', color='k', join='\n', fontsize='xx-small'):
    """
    stats : list
        E.g. ['std', 'wstd']
    join : str
        E.g. whitespace, new line
    """
    s = ''
    if 'std' in stats:
        s += 'std {:.2f}'.format(np.nanstd(data))
    if 'wstd' in stats and err is not None:
        #w = 1./np.array(err)**2
        s += join + 'wstd {:.2f}'.format(compute_wstd(data, err)[0])

    bbox_props = dict(boxstyle="square", fc="w", ec=None, lw=0, alpha=0.7)
    ax.text(x, y, s, ha=ha, va=va, transform=ax.transAxes, fontsize=fontsize, color=color, bbox=bbox_props)
    return ax


def plot_ccfparbasic_servalrvc(data, plotserval=True, shiftserval=True, title='', stats=['std', 'wstd'], **kwargs):
    """
    Plot CCF parameters TS (parameter vs BJD).
    Assume 'RV' and 'BIS' in [km/s] -> transform it to [m/s]
    Assume SERVAL merged with CCF data in `data`.

    stats : List with 'std', 'wstd'
        Show std of the parameter
    """

    # Plot style
    if kwargs is None:
        kwargs = dict()
    kwargs['linestyle'] = kwargs.get('linestyle', 'None')
    kwargs['marker'] = kwargs.get('marker', 'o')

    # Plot
    fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    # BJD
    xsub = 2400000.
    bjd = data['bjd'] - xsub
    xlab = 'BJD - {:.0f}'.format(xsub)

    if plotserval:
        bjds = data['bjd'] - xsub

    # RV
    # - CCF
    if plotserval: lab = 'CCF'
    else: lab = ''
    ax[0].errorbar(bjd, data['rv']*1.e3, yerr=data['rverrabs']*1.e3, ms=5, elinewidth=2, capsize=2, capthick=2, zorder=0, label=lab, **kwargs)
    ax[0].set_ylabel('RV [m/s]')
    printstats(stats, ax[0], data['rv']*1.e3, err=data['rverr']*1.e3)

    # - SERVAL
    if plotserval:
        ys = data['servalrvc']
        lab = 'SERVAL RVC'
        if shiftserval:
            ys = ys - np.nanmedian(ys) + np.nanmedian(data['rv']*1.e3)
            # lab = lab + ' shift'
        l = ax[0].errorbar(bjds, ys, yerr=data['servalrvcerr'], ms=5, elinewidth=2, capsize=2, capthick=2, zorder=1, label=lab, **kwargs)
        printstats(stats, ax[0], data['servalrvc'], err=data['servalrvcerr'], y=0.07, va='bottom', color=l.lines[0].get_color())

    ax[0].legend(fontsize='xx-small', loc='upper left', framealpha=0.7, edgecolor='None')
    # ax[0].legend(fontsize='x-small', loc='upper left', frameon=False)

    # FWHM
    ax[1].errorbar(bjd, data['fwhm'], yerr=data['fwhmerr'], ms=5, elinewidth=2, capsize=2, capthick=2, **kwargs)
    ax[1].set_ylabel('FWHM [km/s]')
    printstats(stats, ax[1], data['fwhm'], err=data['fwhmerr'])

    # Contrast
    ax[2].errorbar(bjd, data['contrast'], yerr=data['contrasterr'], ms=5, elinewidth=2, capsize=2, capthick=2, **kwargs)
    ax[2].set_ylabel('Contrast [%]')
    printstats(stats, ax[2], data['contrast'], err=data['contrasterr'])

    # BIS
    ax[3].errorbar(bjd, data['bis']*1.e3, yerr=data['biserr']*1.e3, ms=5, elinewidth=2, capsize=2, capthick=2, **kwargs)
    ax[3].set_ylabel('BIS [m/s]')
    printstats(stats, ax[3], data['bis']*1.e3, err=data['biserr']*1.e3)

    # General plot
    for a in ax:
        a.minorticks_on()
    ax[0].set_title(title)
    ax[-1].set_xlabel(xlab)

    return fig, ax


def plot_ccfparbasic_servalrvc_separated(data, dataserval=None, shiftserval=True, title='', stats=['std', 'wstd'], **kwargs):
    """
    Plot CCF parameters TS (parameter vs BJD).
    Assume 'RV' and 'BIS' in [km/s] -> transform it to [m/s]
    SERVAL data not merged with CCF data `data` (can have different observations).

    stats : {'std', None}
        Show std of the parameter
    """

    # Plot style
    if kwargs is None:
        kwargs = dict()
    kwargs['linestyle'] = kwargs.get('linestyle', 'None')
    kwargs['marker'] = kwargs.get('marker', 'o')

    # Plot
    fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    # BJD
    xsub = 2400000.
    bjd = data['bjd'] - xsub
    xlab = 'BJD - {:.0f}'.format(xsub)

    if dataserval is not None:
        bjds = dataserval['bjd'] - xsub

    # RV
    # - CCF
    if dataserval is not None: lab = 'CCF {} obs'.format(len(data.index))
    else: lab = ''
    ax[0].errorbar(bjd, data['rv']*1.e3, yerr=data['rverrabs']*1.e3, ms=5, elinewidth=2, capsize=2, capthick=2, zorder=0, label=lab, **kwargs)
    ax[0].set_ylabel('RV [m/s]')
    printstats(stats, ax[0], data['rv']*1.e3, err=data['rverr']*1.e3)

    # - SERVAL
    if dataserval is not None:
        ys = dataserval['servalrvc']
        lab = 'SERVAL RVC {} obs'.format(len(dataserval.index))
        if shiftserval:
            ys = ys - np.nanmedian(ys) + np.nanmedian(data['rv']*1.e3)
            # lab = lab + ' shift'
        l = ax[0].errorbar(bjds, ys, yerr=dataserval['servalrvcerr'], ms=5, elinewidth=2, capsize=2, capthick=2, zorder=1, label=lab, **kwargs)
        printstats(stats, ax[0], dataserval['servalrvc'], err=dataserval['servalrvcerr'], y=0.07, va='bottom', color=l.lines[0].get_color())

    ax[0].legend(fontsize='xx-small', loc='upper left', framealpha=0.7, edgecolor='None')
    # ax[0].legend(fontsize='x-small', loc='upper left', frameon=False)

    # FWHM
    ax[1].errorbar(bjd, data['fwhm'], yerr=data['fwhmerr'], ms=5, elinewidth=2, capsize=2, capthick=2, **kwargs)
    ax[1].set_ylabel('FWHM [km/s]')
    printstats(stats, ax[1], data['fwhm'], err=data['fwhmerr'])

    # Contrast
    ax[2].errorbar(bjd, data['contrast'], yerr=data['contrasterr'], ms=5, elinewidth=2, capsize=2, capthick=2, **kwargs)
    ax[2].set_ylabel('Contrast [%]')
    printstats(stats, ax[2], data['contrast'], err=data['contrasterr'])

    # BIS
    ax[3].errorbar(bjd, data['bis']*1.e3, yerr=data['biserr']*1.e3, ms=5, elinewidth=2, capsize=2, capthick=2, **kwargs)
    ax[3].set_ylabel('BIS [m/s]')
    printstats(stats, ax[3], data['bis']*1.e3, err=data['biserr']*1.e3)

    # General plot
    for a in ax:
        a.minorticks_on()
    ax[0].set_title(title)
    ax[-1].set_xlabel(xlab)

    return fig, ax


def plot_ccfrv(data, title='', stats=['std', 'wstd'], **kwargs):
    """
    stats : {'std', None}
        Show std of the parameter
    """

    # Plot style
    if kwargs is None:
        kwargs = dict()
    kwargs['linestyle'] = kwargs.get('linestyle', 'None')
    kwargs['marker'] = kwargs.get('marker', 'o')

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

    # BJD
    xsub = 2400000.
    bjd = data['bjd'] - xsub
    xlab = 'BJD - {:.0f}'.format(xsub)

    # RV CCF
    y = data['rv']*1.e3
    ax.errorbar(bjd, y, yerr=data['rverrabs']*1.e3, ms=5, elinewidth=2, capsize=2, capthick=2, **kwargs)
    ax.set_ylabel('RV [m/s]')
    printstats(stats, ax, data['rv']*1.e3, err=data['rverr']*1.e3)

    # General plot
    ax.minorticks_on()
    ax.set_title(title)
    ax.set_xlabel(xlab)

    return fig, ax


def plot_ccfrv_servalrvc(data, shiftserval=True, title='', stats=['std', 'wstd'], **kwargs):
    """
    stats : {'std', None}
        Show std of the parameter
    """

    # Plot style
    if kwargs is None:
        kwargs = dict()
    kwargs['linestyle'] = kwargs.get('linestyle', 'None')
    kwargs['marker'] = kwargs.get('marker', 'o')

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

    # BJD
    xsub = 2400000.
    bjd = data['bjd'] - xsub
    xlab = 'BJD - {:.0f}'.format(xsub)

    # RV
    # - CCF
    y = data['rv']*1.e3
    lab = 'CCF'
    ax.errorbar(bjd, y, yerr=data['rverrabs']*1.e3, label=lab, ms=5, elinewidth=2, capsize=2, capthick=2, zorder=0, **kwargs)
    ax.set_ylabel('RV [m/s]')
    printstats(stats, ax, data['rv']*1.e3, err=data['rverr']*1.e3)

    # - SERVAL
    ys = data['servalrvc']
    lab = 'SERVAL RVC'
    if shiftserval:
        ys = ys - np.nanmedian(ys) + np.nanmedian(data['rv']*1.e3)
        # lab = lab + ' shift'
    l = ax.errorbar(bjd, ys, yerr=data['servalrvcerr'], label=lab, ms=5, elinewidth=2, capsize=2, capthick=2, zorder=1, **kwargs)
    printstats(stats, ax, data['servalrvc'], err=data['servalrvcerr'], y=0.07, va='bottom', color=l.lines[0].get_color())

    ax.legend(fontsize='xx-small', loc='upper left', framealpha=0.7, edgecolor='None')

    # General plot
    ax.minorticks_on()
    ax.set_title(title)
    ax.set_xlabel(xlab)

    return fig, ax


def plot_ccfrv_servalrvc_diff(data, shiftserval=True, title='', stats=['std', 'wstd'], **kwargs):
    """
    stats : {'std', None}
        Show std of the parameter
    """

    # Plot style
    if kwargs is None:
        kwargs = dict()
    kwargs['linestyle'] = kwargs.get('linestyle', 'None')
    kwargs['marker'] = kwargs.get('marker', 'o')

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios':[3, 1]})

    # BJD
    xsub = 2400000.
    bjd = data['bjd'] - xsub
    xlab = 'BJD - {:.0f}'.format(xsub)

    # RV
    # - CCF
    y = data['rv']*1.e3
    lab = 'CCF RV'
    ax[0].errorbar(bjd, y, yerr=data['rverrabs']*1.e3, label=lab, ms=5, elinewidth=2, capsize=2, capthick=2, zorder=0, **kwargs)
    ax[0].set_ylabel('RV [m/s]')
    printstats(stats, ax[0], data['rv']*1.e3, err=data['rverr']*1.e3)

    # - SERVAL
    ys = data['servalrvc']
    lab = 'SERVAL RVC'
    if shiftserval:
        ys = ys - np.nanmedian(ys) + np.nanmedian(data['rv']*1.e3)
        # lab = lab + ' shift'
    l = ax[0].errorbar(bjd, ys, yerr=data['servalrvcerr'], label=lab, ms=5, elinewidth=2, capsize=2, capthick=2, zorder=1, **kwargs)
    printstats(stats, ax[0], data['servalrvc'], err=data['servalrvcerr'], y=0.07, va='bottom', color=l.lines[0].get_color())

    ax[0].legend(fontsize='xx-small', loc='upper left', framealpha=0.7, edgecolor='None')

    # Difference residuals
    diff = y - ys
    ax[1].plot(bjd, diff, **kwargs)
    ax[1].hlines(0, np.nanmin(bjd), np.nanmax(bjd), colors='.6', linestyle='dashed')
    ax[1].set_ylabel('Diff\n[m/s]')
    printstats(stats, ax[1], diff)

    # General plot
    for a in ax:
        a.minorticks_on()
    ax[0].set_title(title)
    ax[-1].set_xlabel(xlab)

    return fig, ax
