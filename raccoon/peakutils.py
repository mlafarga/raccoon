#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import sys

from lmfit import Model
import numpy as np
import pandas as pd
from progress.bar import Bar
import scipy.signal

###############################################################################


# Find peaks (absorption lines)
# -----------------------------

# Find maxima closest to each side of a specifit minimum

def min_find_maxabs(y, imin=None):
    """
    Find the absolute maxima at each side of the minimum in `y`.

    The minimum position (array index) can be indicated with `imin`. If None (default), find absolute minimum in `y`.

    Returns
    -------
    imax1, imax2 : int
        Indices of the maximum at the left side and at the right side.
    imin : int
        Index of the minimum.
    """
    # Minimum
    if imin is None:
        imin = np.nanargmin(y)
    # Absolute maximum left side
    if imin != 0:
        imax1 = np.nanargmax(y[:imin])
    else:
        imax1 = imin
    # Absolute maximum right side
    if imin != len(y):
        imax2 = imin + np.nanargmax(y[imin:])
    else:
        imax2 = imin
    return imax1, imax2, imin


def min_find_maxclosest(y, imin=None):
    """Find the maxima closest the minimum in `y`, one at each side.

    The minimum position can be indicated with `imin`. If None (default), find absolute minimum in `y`.

    Returns
    -------
    imax1, imax2 : int
        Indices of the maximum at the left side and at the right side.
    imin : int
        Index of the minimum.
    """
    # Minimum
    if imin is None:
        imin = np.nanargmin(y)
    # All minima and maxima
    limin, limax1, limax2 = find_abspeaks(y, method='custom')
    # Maxima closest to minimum
    i = np.where(limin == imin)[0][0]  # locate our minimum in the all minima list
    imax1, imax2 = limax1[i], limax2[i]  # closest maxima
    return imax1, imax2, imin


# Find minima and maxima

def idxmin_custom(f):
    """Find all minima in spectrum. Custom method.

    Parameters
    ----------
    f : 1d array-like of numbers
        Flux values of a single order.

    Returns
    -------
    idxmin : 1d array of int
        Indices of all the flux minima in `f`.
    """
    imin = []
    i = 1
    while i < (len(f)-1):
        if f[i] < f[i-1]:
            if f[i] < f[i+1]:  # => f[i] = min
                imin.append(i)
                i = i + 1
            elif f[i] > f[i+1]:  # => f[i] != min
                i = i + 1
            elif f[i] == f[i+1]:  # check next points
                for j in range(1, len(f)-1-i):
                    if f[i] < f[i+2]:  # => f[i] = min
                        imin.append(i)
                        i = i + 2
                        break
                    elif f[i] > f[i+2]:  # => f[i] != min
                        i = i + 2
                        break
                    elif f[i] == f[i+2]:  # check next point
                        i = i + 1
        else:
            i = i + 1
    imin = np.array(imin)
    return imin


def idxminends_custom(f, imin):
    """
    Find "ends of the minima" in spectrum `f`, i.e. maxima at each side of the minimum.
    Minima in `imin` must be consecutive. If there are telluric regions (broad regions with no data), the maxima of the lines in the limits of these regions will not be correct, so it will be necessary to remove the lines at the telluric regions limits.

    Parameters
    ----------
    f : 1d array-like of numbers
        Flux values of a single order.
    imin

    Returns
    -------
    imax1 : array, int
    imax2 : array, int
    """
    imax1, imax2 = [], []
    for i in imin:
        if i == imin[0]:  # First minimum
            b = i
            while f[b] < f[b-1] and b != 0:
                b = b - 1
            imax1.append(b)
            r = i
            while f[r] < f[r+1] and r != len(f)-2:
                r = r + 1
            imax2.append(r)
        else:
            b = imax2[-1]
            imax1.append(b)
            r = i
            while f[r] < f[r+1] and r != len(f)-2:
                r = r + 1
            imax2.append(r)
    imax1, imax2 = np.array(imax1), np.array(imax2)
    return imax1, imax2


# def idxmin_find_peaks(f):
#     """Find all minima in spectrum using scipy's function `find_peaks`.
#     """
#     imin = scipy.signal.find_peaks(-f)
#     return imin


# Find minima and maxima in spectrum

def find_abspeaks(f, method='custom'):
    """Find minima and maxima of spectrum `f`."""
    if method == 'custom':
        imin = idxmin_custom(f)
        imax1, imax2 = idxminends_custom(f, imin)

    elif method == 'scipy.signal.find_peaks':
        sys.exit('Find minima and maxima with scipy.signal.find_peaks. Not implemented yet!')
        imin = scipy.signal.find_peaks(-f)
        imax = scipy.signal.find_peaks(f)
        # Separate imax
        imax1 = [i for i in range(len(imax)) if i % 2 == 0]
        imax2 = [i for i in range(len(imax)) if i % 2 == 1]
        # Does this work for all cases?

    elif method == 'derivative':
        sys.exit('Find minima and maxima by computing the spectrum derivative. Not implemented yet!')

    return imin, imax1, imax2


def find_abspeaks_echelle(f, method='custom', ords_use=None):
    """
    """
    nord = len(f)
    ords = np.arange(0, nord, 1)
    if ords_use is None: ords_use = ords

    imin, imax1, imax2 = [[]]*nord, [[]]*nord, [[]]*nord
    for o in ords:
        if o in ords_use:
            imin[o], imax1[o], imax2[o] = find_abspeaks(f[o], method=method)
    return imin, imax1, imax2

###############################################################################


# Peaks and lines

def b_overlap_a(a, b, alab='a', blab='b', verb=False):
    """Check if there is any value in `b` inside the range of the values in `a`.

    Only checks if there is overlap, but not how (e.g. do not tell if all values of b inside a, or just from one side).

    Examples
    --------
    a: '     |---|     '
    b: '|--|           ' -> Outside
    b: '|------|       ' -> Inside
    b: '     |--|      ' -> Inside
    b: '       |------|' -> Inside
    b: '           |--|' -> Outside
    """
    if b[0] > a[-1]:
        if verb: print('All values of {} outside of the range of {} ({}[0] > {}[-1])'.format(blab, alab, blab, alab))
        ret = False
    elif b[-1] < a[0]:
        if verb: print('All values of {} outside of the range of {} ({}[-1] < {}[0])'.format(blab, alab, blab, alab))
        ret = False
    else:
        if verb: print('Some values of {} inside the range of {}.')
        ret = True
    return ret


def find_abspeak_list(w, f, wlines, wlines_idx=None, method='custom', imin=None, imax1=None, imax2=None, returndf=True, verb=False):
    """Find the minimum peaks in the spectrum `w`, `f` closest to the wavelength positions in the list `wlines`.

    Parameters
    ----------
    w, f : 1d array-like
    wlines : 1d array-like
    wlines_idx : 1d array-like
        Identifier for the lines in `wlines`.
        If lines in a pandas dataframe df, `wlines=df['w'].values` and `wlines_idx=df.index`
    method : {'custom', 'find_peaks'}
        Method used to locate the peaks in the spectrum `w`, `f`
    imin, imax1, imax2 : 1d array-like
        If minima already computed, can provide the indices with these arguments. I.e. `imin` indices of the minima in `f`, and `imax1` and `imax2` indices of the maxima closest to the minima.
    returndf : bool
        Whether to return the data in a pandas dataframe or a dict.

    Returns
    -------
    data : dict or pandas dataframe
        For each line in `wlines`, return the values of the origina lines in `wlines`, the index of the peak closest in `imin`, the pixel closest in `w` and the wavelenght of `w` corresponding to these indices.

    """

    data = {
        'w_list': wlines,
        'ipeak_spec': np.nan,
        'ipix_spec': np.nan,
        'w_spec': np.nan,
    }

    # Check that the input arrays are not all nan
    # If they are, return values are nan
    input_nan = False
    for inp in [(wlines, 'wlines'), (w, 'w'), (f, 'f')]:
        if np.isnan(inp[0]).all():
            if verb: print('Input {} only contains nan'.format(inp[1]))
            input_nan = True
    if input_nan:
        if returndf: data = pd.DataFrame.from_dict(data, orient='columns')
        return data

    # Check if lines in `wlines` are inside spectrum `w` range (assume lines and spectrum sorted)
    # If not, return values are nan
    if not b_overlap_a(w, wlines, alab='w', blab='wlines'):
        if returndf: data = pd.DataFrame.from_dict(data, orient='columns')
        return data

    # All checks passed

    # Identifier for the lines in `wlines`
    if wlines_idx is None: wlines_idx = np.arange(0, len(wlines), 1)

    # If not provided, find absorption peak minimum and maxima
    if (imin is None) and (imax1 is None) and (imax2 is None):
        imin, imax1, imax2 = find_abspeaks(f, method=method)

    # Find the minimum peaks in spectrum `w`, `f` closest to the lines in `wlines`
    ipeak = []
    for wl in wlines:
        # Find minimum in `w` closest to `wl`
        ipeak.append(np.argmin(np.abs(w[imin] - wl)))  # Index of the peak: w[imin[ipeak]]
    ipix = imin[ipeak]  # Index of the spectrum pixel w[ipix]

    data = {
        'w_list': wlines,
        'ipeak_spec': ipeak,
        'ipix_spec': ipix,
        'w_spec': w[ipix],  # Same as: w[imin[ipeak]]
    }

    if returndf: data = pd.DataFrame(data, index=wlines_idx)
    # if returndf: data = pd.DataFrame.from_dict(data, orient='columns')
    return data


def find_abspeak_list_echelle(w, f, wlines, wlines_idx=None, method='custom', imin=None, imax1=None, imax2=None, returndf=True, ords_use=None, verb=False):

    nord = len(w)
    ords = np.arange(0, nord, 1)
    if ords_use == None: ords_use = ords

    data = [[]]*nord
    for o in ords:
        if verb: print(o)
        if o in ords_use:
            # Find lines inside w[o] range
            i1 = np.argmin(np.abs(wlines - w[o][0]))
            i2 = np.argmin(np.abs(wlines - w[o][-1]))
            # Identify minima in spectrum
            data[o] = find_abspeak_list(w[o], f[o], wlines[i1:i2], wlines_idx=wlines_idx[i1:i2], method='custom', imin=imin[o], imax1=imax1[o], imax2=imax2[o], returndf=True, verb=verb)
        else:
            # Force return nan
            data[o] = find_abspeak_list([np.nan], [np.nan], [np.nan], returndf=True, verb=verb)

    return data

###############################################################################


# Fit peaks

def gaussian(x, amp=1, cen=0, wid=1, shift=0):
    """Gaussian function: `G(x) = shift + amp * e**(-(x-cen)**2 / (2*wid**2))`

    Function properties
    -------------------
    - Full width at half maximum: `FWHM = 2 * sqrt(2*ln(2)) * wid`
    - Center: `cen`
    - Maximum value: `shift + amp`
    - Minimum value: `shift`
    """
    return shift + amp * np.exp(-(x-cen)**2 / (2*wid**2))


def gaussian_fwhm(wid=1):
    return 2 * np.sqrt(2*np.log(2)) * wid


def gaussian_fwhmerr(wid=1, widerr=np.nan):
    fwhm = 2 * np.sqrt(2*np.log(2)) * wid
    fwhmerr = 2 * np.sqrt(2*np.log(2)) * widerr
    return fwhm, fwhmerr


def gaussian_contrasterr(amp, shift, amperr=np.nan, shifterr=np.nan):
    contrast = - (amp/shift) * 100.
    contrasterr = 100 / shift**2 * np.sqrt((shift*amperr)**2 + (amp * shifterr)**2)
    return contrast, contrasterr


def gaussian_minmax(shift=0, amp=1):
    gmin = shift
    gmax = shift + amp
    return gmin, gmax


def fit_gaussian_peak(w, f, amp_hint=-0.5, cen_hint=None, wid_hint=0.01, shift_hint=0.8, minmax='min'):
    """Fit a single Gaussian to a spectrum line.

    Gaussian G(x)=shift+amp*e^(-(x-cen)^2/(2*wid^2))

    Uses lmfit package.

    Parameters
    ----------
    w, f : 1d arrays
        Spectrum range where the minimum is located.
    amp_hint : float
        Amplitude value to be used as 1st guess when performing the fit.
    cen_hint : float
        Wavelength of the minimum location, to be used as 1st guess when fitting. If None, the mean value of `w` is used.
    wid_hint : float
    shift_hint : float
    minmax = str
        Specifies if the peak to be fitted is a minimum or a maximum, to know if the gaussian amplitude parameter `amp` must be negative or positive, respectively. Default: 'min'.

    Returns
    -------
    lmfit results object

    """

    def gaussian(x, amp=1, cen=0, wid=1, shift=0):
        return shift + amp * np.exp(-(x-cen)**2 / (2*wid**2))

    # Fit model and parameters
    mod = Model(gaussian)

    # Amplitude `amp`
    if minmax == 'min': mod.set_param_hint('amp', value=amp_hint, max=0.)
    elif minmax == 'max': mod.set_param_hint('amp', value=amp_hint, min=0.)
    # Center `cen`
    if cen_hint is None: cen_hint = np.mean(w)
    mod.set_param_hint('cen', value=cen_hint)
    # Width `wid`
    mod.set_param_hint('wid', value=wid_hint, min=0.)
    # Shift in the y-axis `shift`
    mod.set_param_hint('shift', value=shift_hint)
    # mod.set_param_hint('fwhm', expr='2.35482004503*wid')
    # mod.set_param_hint('height', expr='shift+amp')

    gfitparams = mod.make_params()

    # Fit
    lmfitresult = Model(gaussian).fit(f, x=w, params=gfitparams)

    return lmfitresult


def fit_gaussian_spec(w, f, imin, imax1, imax2, amp_hint=-0.5, cen_hint=None, wid_hint=0.01, shift_hint=0.8, minmax='min', nfitmin=None, returntype='pandas', barmsg=''):
    """Fit `len(imin)` peaks in the spectrum `w`, `f`.

    Gaussian G(x)=shift+amp*e^(-(x-cen)^2/(2*wid^2)).

    Uses `fit_gaussian_peak`.

    Parameters
    ----------
    amp_hint : array or float
        Amplitude value(s) to be used as 1st guess when performing the fit. If it is a float, the same value is used for all the peaks. If it is an array, it must contain a value for each peak in w[imin], i.e. len(amp_hint) == len(imin).
    cen_hint : array, float or None
        Wavelength(s) of the minimum location, to be used as 1st guess when fitting. If it is a float, the same value is used for all the peaks. If it is an array, it must contain a value for each peak, e.g. `w[imin]`. If None, the mean value of `w` is used.
    wid_hint : array or float
    shift_hint : array or float
    minmax = str
        Specifies if the peak to be fitted is a minimum or a maximum, to know if the gaussian amplitude parameter `amp` must be negative or positive, respectively. Default: 'min'.
    returntype : {'pandas', 'lmfit'}
        Type of returned value. If 'pandas', return a pandas dataframe with the most important fit parameters of each peak. If 'lmfit', return a list with the output of the lmfit fit function.
    }
    """
    # Total number of peaks
    npeak = len(imin)

    # Parameters
    param_names = ['amp', 'cen', 'shift', 'wid']
    # Parameter 1st guesses
    hints = {'amp': amp_hint, 'cen': cen_hint, 'shift': shift_hint, 'wid': wid_hint}

    # Minimum number of datapoints that a peak must have in order to be fitted
    if nfitmin is None: nfitmin = len(param_names)  # Must be equal or larger than the number of parameters
    # Select peaks with enough datapoints
    mask = np.asarray([len(w[imax1[i]:imax2[i]+1]) >= nfitmin for i in range(npeak)])
    # imin_fit, imax1_fit, imax2_fit = [], [], []
    # for i in range(npeak): # for each peak
    #    if len(w[imax1[i]:imax2[i]+1]) >= nfitmin:
    #        imin_fit.append(imin[i])
    #        imax1_fit.append(imax1[i])
    #        imax2_fit.append(imax2[i])
    # # Number of peaks to fit
    # npeak_fit = len(imin_fit)

    # Check if parameter hints are floats or arrays. If float (or None, for the parameter `cen`), convert them to an array of length npeak containing the float value (or the None value). If array, check that its length is equal to the number of peaks.
    for key, val in hints.items():
        if isinstance(val, (int, float)) or val is None:
            hints[key] = [val]*npeak  # Same value for all peaks
        elif isinstance(val, (list, tuple, np.ndarray)):
            if len(val) != npeak: raise ValueError('`{}_hint` must have the same length as the number of peaks, {}'.format(key, npeak))

    # Same for parameter `minmax`
    if isinstance(minmax, str):
        minmax = [minmax]*npeak  # Same value for all peaks
    elif isinstance(minmax, (list, tuple, np.ndarray)):
        if len(minmax) != npeak: raise ValueError('`minmax` must have the same length as the number of peaks, {}'.format(npeak))

    # Fit
    dataout = [[]]*npeak
    for i in Bar(barmsg, max=npeak).iter(range(npeak)):  # for each peak
        # if vb: print('  {}/{}'.format(i+1, npeak))
        # If enough datapoints
        if mask[i]:
            lmfitresult = fit_gaussian_peak(w[imax1[i]:imax2[i]+1], f[imax1[i]:imax2[i]+1], amp_hint=hints['amp'][i], cen_hint=hints['cen'][i], wid_hint=hints['wid'][i], shift_hint=hints['shift'][i], minmax=minmax[i])

            # Get fit results
            if returntype == 'pandas':
                dataout[i] = {}
                for p in param_names:
                    dataout[i][p] = lmfitresult.params[p].value
                    dataout[i][p+'err'] = lmfitresult.params[p].stderr
                dataout[i]['redchi2'] = lmfitresult.redchi
            elif returntype == 'lmfit':
                dataout[i] = lmfitresult

        # Else if cannot fit because not enough datapoints
        else:
            if returntype == 'pandas':
                dataout[i] = {}
                for p in param_names:
                    dataout[i][p] = np.nan
                    dataout[i][p+'err'] = np.nan
                dataout[i]['redchi2'] = np.nan
            elif returntype == 'lmfit':
                dataout[i] = np.nan

    if returntype == 'pandas':
        dataout = pd.DataFrame(dataout, index=imin)
        # Add more info
        dataout['fwhm'] = 2 * np.sqrt(2*np.log(2)) * dataout['wid']
        dataout['fwhmerr'] = np.nan
        dataout['imin'] = imin
        dataout['imax1'] = imax1
        dataout['imax2'] = imax2
        dataout['wmin'] = w[imin]
        dataout['wmax1'] = w[imax1]
        dataout['wmax2'] = w[imax2]
        dataout['fmin'] = f[imin]
        dataout['fmax1'] = f[imax1]
        dataout['fmax2'] = f[imax2]

        columnsorder = ['amp', 'amperr', 'cen', 'cenerr', 'shift', 'shifterr', 'wid', 'widerr', 'fwhm', 'fwhmerr', 'redchi2', 'imin', 'imax1', 'imax2', 'wmin', 'wmax1', 'wmax2', 'fmin', 'fmax1', 'fmax2']
        dataout.index.names = ['imin']
        return dataout[columnsorder]

    elif returntype == 'lmfit':
        return dataout
