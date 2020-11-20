#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import pickle
import sys

from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import maximum_filter1d

###############################################################################


# Wavelength vacuum/air

def wvac2air(w):
    """Transform vacuum wavelength to air wavelength.
    Formula from: Ciddor 1996, Applied Optics 62, 958.

    w : float or array-like
        Vacuum wavelength to be transformed to air, in A. If array-like, w sorted in increasing or decreasing order.
    """
    scalar = False
    if isinstance(w, (int, float)):
        w = [w]
        scalar = True
    w = np.array([w])
    wair = w.copy()

    mask = w > 2000.  # Modify only wavelength above 2000 A

    s2 = (1e4/w[mask])**2
    f = 1.+0.05792105/(238.0185-s2)+0.00167917/(57.362-s2)
    wair[mask] = w[mask]/f
    return wair[0][0] if scalar else wair[0]


def wair2vac(w):
    """Transform air wavelength to vacuum wavelength.
    Formula from: Ciddor 1996, Applied Optics 62, 958.

    w : float or array-like
        Air wavelength to be transformed to vacuum, in A. If array-like, w sorted in increasing or decreasing order.
    """
    scalar = False
    if isinstance(w, (int, float)):
        w = [w]
        scalar = True
    w = np.array([w])
    wvac = w.copy()

    mask = w > 2000.  # Modify only wavelength above 2000 A

    s2 = (1e4/w[mask])**2
    f = 1.+0.05792105/(238.0185-s2)+0.00167917/(57.362-s2)
    wvac[mask] = w[mask]*f
    return wvac[0][0] if scalar else wvac[0]

###############################################################################


# Clean

def remove_nan(*args):
    """
    *args: 1d arrays of the same length
    """
    # Check input arrays have the same wavelegth
    # ...

    # Find bad pixels: nan, null or negative wavelengths
    mask = []

    mask = np.ones_like(args[0], dtype=bool)
    for a in args:
        for i in np.arange(len(a)):
            # NaN, null
            if not np.isfinite(a[i]):
                mask[i] = False
                continue
            # Negative values
            elif a[i] < 0.:
                mask[i] = False

    # Remove datapoints for which mask == False
    args_new = []
    for a in args:
        args_new.append(a[mask])
    return args_new, mask


def remove_nan_echelle(*args, **kwargs):
    """
    Parameters
    ----------
    *args: nd arrays of the same dimension
    **kwargs: `ords_use`, `returntype`
    ords_use
    returntype : str
        For the orders skipped (orders not in `ords_use`), return
        - `returntype` = 'original': the original spectrum
        - `returntype` = 'empty': empty list
        - `returntype` = 'nan': array with the same size as the original one but filled with np.nan
    """
    # Check input arrays have the same wavelegth
    # ...

    nord = len(args[0])
    ords = np.arange(0, nord, 1)
    narg = len(args)

    ords_use = kwargs.pop('ords_use', None)  # Return 'ords_use' value given in kwargs, or None if 'ords_use' not found in kwargs
    if ords_use is None: ords_use = ords
    returntype = kwargs.pop('returntype', 'original')  # Return `returntype` value given in kwargs, or 'original' if `returntype` not found in kwargs
    if returntype not in ['original', 'empty', 'nan']: raise ValueError('Invalid value for `returntype`: {}. Should be one of the following: original, empty or nan'.format(returntype))
    if kwargs: raise TypeError('Unexpected positional arguments', kwargs)

    # Find bad pixels: nan, null or negative wavelengths
    mask = [[]]*nord
    for o in ords:
        if o in ords_use:
            mask[o] = np.ones_like(args[0][o], dtype=bool)
            for a in args:
                for i in np.arange(len(a[o])):
                    # NaN, null
                    if not np.isfinite(a[o][i]):
                        mask[o][i] = False
                        continue
                    # Negative values
                    elif a[o][i] < 0.:
                        mask[o][i] = False
        else:
            # Return original array
            if returntype == 'original':
                mask[o] = np.ones_like(args[0][o], dtype=bool)
            # Return empty array
            elif returntype == 'empty':
                mask[o] = np.zeros_like(args[0][o], dtype=bool)
            # Return nan array
            elif returntype == 'nan':
                mask[o] = np.ones_like(args[0][o])*np.nan

    # Remove bad pixels
    args_new = [[]]*narg
    for j, a in enumerate(args):
        args_new[j] = [[]]*nord
        for o in ords:
            if o in ords_use:
                args_new[j][o] = a[o][mask[o]]
            else:
                if returntype == 'nan':
                    args_new[j][o] = mask[o]
                else:
                    args_new[j][o] = a[o][mask[o]]
    return args_new, mask

###############################################################################


# Smooth

def conv(f, kerneltype, kernelwidth, boundary='extend'):
    """Convolve spectrum with a kernel to smooth it.

    Uses kernels and convolution function from `astropy`. See more kernels in `astropy.convolution`.

    Parameters
    ----------
    f : array
    kerneltype : str [['Gaussian','Box']]
        If 'Box', use `Box1DKernel(width)`, where `width` is the width of the filter kernel.
        If 'Gaussian', use `Gaussian1DKernel`, where `width` is the standard deviation of the Gaussian kernel. Default size of the kernel array: 8*stddev. Can change with `x_size` parameter.
    kernelwidth : float
        See `kerneltype`.
    """
    if kerneltype == 'box': kernel = Box1DKernel(kernelwidth)
    elif kerneltype == 'gaussian': kernel = Gaussian1DKernel(kernelwidth)
    fconv = convolve(f, kernel, boundary=boundary)
    return fconv


def conv_echelle(f, kerneltype, kernelwidth, boundary='extend', ords_use=None, returnfill=True):
    """Same as `conv` but with several spectral orders.

    Parameters
    ----------
    returnfill : bool
        For the orders skipped (orders not in `ords_use`), return the original spectrum, instead of an empty array.
    """
    nord = len(f)
    ords = np.arange(0, nord, 1)
    if ords_use is None: ords_use = ords

    fconv = [[]]*nord
    for o in ords:
        if o in ords_use:
            fconv[o] = conv(f[o], kerneltype, kernelwidth, boundary='extend')
        else:
            if returnfill:
                fconv[o] = f[o]

    return fconv

###############################################################################


# Continuum fitting

def filtermed(f, medfiltsize=9):  # medfiltfunc='ndimage'
    """Apply median filter to the spectrum to smooth out single-pixel deviation
    Median filter from scipy.ndimage.filter faster than median filter from scipy.signal.
    """
    # # Median filter to smooth out single-pixel deviations
    # if medfiltfunc == 'ndimage': f_medfilt = median_filter(f, size=medfiltsize)
    # elif medfiltfunc == 'signal': f_medfilt = medfilt(f, kernel_size=medfiltsize)
    f_medfilt = median_filter(f, size=medfiltsize)
    return f_medfilt


def filtermax(f, maxfiltsize=10):
    """Apply  maximum filter to the spectrum to ignore deeper fluxes of absorption lines."""
    # Maximum filter to ignore deeper fluxes of absorption lines
    f_maxfilt = maximum_filter1d(f, size=maxfiltsize)
    # Find points selected by maximum filter
    idxmax = np.array([i for i in range(len(f)) if f[i]-f_maxfilt[i] == 0.])

    return f_maxfilt, idxmax


def fitcontinuum(w, f, medfiltsize=9, maxfiltsize=10, fitfunc='poly', polyord=3, splsmooth=None, spldegree=3):
    """
    spldegree : int, <=5
        Degree of the smoothing spline. Default: 3, cubic spline.
    """

    # Select continuum pixels
    f_medfilt = filtermed(f, medfiltsize=medfiltsize)
    f_maxfilt, idxmax = filtermax(f_medfilt, maxfiltsize=maxfiltsize)

    # Fit function to selected points
    if fitfunc == 'poly':
        #--n = 1.
        fitpar = np.polyfit(w[idxmax], f[idxmax], polyord)
        Cont = np.poly1d(fitpar) # Function
    elif fitfunc == 'spl':
        n = np.nanmax(f) # `UnivariateSpline` has problems with large values
        # print(w[idxmax])
        # print(f[idxmax]/n)
        Cont_n = UnivariateSpline(w[idxmax], f[idxmax]/n, k=spldegree, s=splsmooth) #Function
        #--Cont = UnivariateSpline(w[idxmax], f[idxmax], k=spldegree, s=splsmooth) #Function
        def Cont(x, Cont_n=Cont_n, n=n):
            return Cont_n(x)*n
        fitpar = None

    #--c = Cont(w)*n # Array
    c = Cont(w) # Array
    fc = np.array(f/c)

    return fc, c, Cont, f_medfilt, f_maxfilt, idxmax, fitpar


def fitcontinuum_echelle(w, f, medfiltsize=9, maxfiltsize=10, fitfunc='poly', polyord=3, splsmooth=None, spldegree=3, ords_use=None, returnfill=True):
    """Same as `fitcontinuum` but with several spectral orders.

    Parameters
    ----------
    returnfill : bool
        For the orders skipped (orders not in `ords_use`), return the original spectrum, instead of an empty array.
    """
    nord = len(w)
    ords = np.arange(0, nord, 1)
    if ords_use is None: ords_use = ords

    fc, c, Cont, f_medfilt, f_maxfilt, idxmax, fitpar = [[]]*nord, [[]]*nord, [[]]*nord, [[]]*nord, [[]]*nord, [[]]*nord, [[]]*nord
    for o in ords:
        if o in ords_use:
            fc[o], c[o], Cont[o], f_medfilt[o], f_maxfilt[o], idxmax[o], fitpar[o] = fitcontinuum(w[o], f[o], medfiltsize=medfiltsize, maxfiltsize=maxfiltsize, fitfunc=fitfunc, polyord=polyord, splsmooth=splsmooth, spldegree=spldegree)
        else:
            if returnfill:
                fc[o] = f[o]
                c[o] = np.ones_like(f[o])
                Cont[o] = lambda x: np.ones_like(x)  # Returns array of 1

    return fc, c, Cont, f_medfilt, f_maxfilt, idxmax, fitpar

###############################################################################


# Instrumental broadening

def conv_gauss_custom(x, y, fwhm, dwindow=2):
    """
    Compute the convolution of the input data (`x`, `y`) with a Gaussian of a certain width.
    The width of the Gaussian can be the same number for all datapoints or can be given by an array which gives a fwhm for datapoints (see below).

    Parameters
    ----------
    x, y : 1D array-like
        Input data.
    fwhm : float or 1D array-like with same length as input data
        Width of the Gaussian function to use as kernel. If a single number, the same width will be used for all datapoints. If an array, a different width will be used for each datapoint.
    dwindow : int
        Number of fwhms to define the half-window of data to use as the kernel size. I.e. when computing the convolution of the datapoint x_i, the Gaussian applied as kernel will be defined from `x_i - dwindow*fwhm` until `x_i + dwindow*fwhm`.
    """
    # fwhm = sigma * 2 * np.sqrt(2 * np.log(2))

    # Check if fwhm is a number or a list
    if isinstance(fwhm, (int, float)):
        # If fwhm is a number, make an array with fwhm in each entry
        fwhm = np.ones_like(x) * fwhm
    else:
        # Check fwhm has same dimensions as x
        if len(fwhm) != len(x):
            sys.exit('Array `fwhm` has different length than `x`: len(fwhm)={}, len(x)={}'.format(len(fwhm), len(x)))

    # Number of total datapoints
    nx = len(x)

    # -----------------------

    # For each datapoint define a "bin" or "pixel"
    # E.g. for the datapoint x_3:
    # - Bin center: value of the datapoint: x_3
    # - Bin left edge: half the distance between the current datapoint and the previous one: x_3 - (x_3 - x_2) * 0.5
    # - Bin right edge: half the distance between the current datapoint and the next one: x_3 + (x_4 - x_3) * 0.5

    # Distances between center of each bin
    bin_distance = x[1:] - x[:-1] # length = len(x) - 1
    # Define left/right edge of each bin as half the distance to the bin previous/next to it
    bin_edgesmiddle = x[:-1] + 0.5 * bin_distance # middle points
    bin_edgesfirst = x[0] - 0.5 * bin_distance[0] # first point
    bin_edgeslast = x[-1] + 0.5 * bin_distance[-1] # last point
    edges = np.concatenate(([bin_edgesfirst], bin_edgesmiddle, [bin_edgeslast]), axis=0) # length = len(x) + 1

    # Width of each bin
    #  If the input array x is equally spaced, `bin_width` will be equal to `bin_distance`
    bin_width = edges[1:] - edges[:-1] # length = len(x)

    # -----------------------

    # Convert FWHM from wavelength units to bins -> Number of bins per FWHM
    fwhm_bin = fwhm / bin_width
    # Round number of bins per FWHM
    nbins = np.ceil(fwhm_bin) #npixels

    ## Convert sigma from wavelength units to bins -> Number of bins per sigma
    #sigma_bin = sigma / bin_width
    ## Round number of bins per sigma
    #nbins = np.ceil(sigma_bin) #npixels

    # -----------------------

    yconv = np.zeros_like(x)
    for i, x_i in enumerate(x):

        # Slow method -> THIS IS WHAT MAKES THE OTHER FUNCTION SLOW!
        # # Select kernel window
        # dwindow = 2 * fwhm #2 * fwhm
        # x1 = (np.argmin(np.abs(x - (x_i - dwindow))))
        # x2 = (np.argmin(np.abs(x - (x_i + dwindow))))
        # irang = slice(x1, x2+1)

        # Number of pixels at each side of x_i:
        dx = dwindow * nbins[i] * 0.5
        i1 = int(max(0, i - dx))
        i2 = int(min(nx, i + dx + 1))
        irang = slice(i1, i2 + 1)

        # Gaussian kernel
        kernel = 1./(np.sqrt(2*np.pi)*fwhm[i]) * np.exp(- ((x[irang] - x_i)**2) / (2 * fwhm[i]**2))
        kernel = kernel / np.sum(kernel)

        # Convolve
        yconv[i] = np.sum(y[irang] * kernel)

    return yconv


def spec_conv_gauss_custom(x, y, resolution=None, fwhm=None, dwindow=2, verb=True):
    """
    Compute the convolution of the spectrum (`x`, `y`) with a Gaussian function.
    The width of the Gaussian function can be given by a fixed fwhm or by the target resolution of the output (see below).

    Parameters
    ----------
    x, y : 1D array-like
        Input data.
    resolution : float
        Resolving power of the new data.
        The `fwhm` of the kernel to be applied to datapoint x_i is computed as
            fwhm = w_i / resolution
            fwhm = fwhm / (2 * np.sqrt(2 * np.log(2)))
        If present, overrides fwhm (see below).
    fwhm : float
        Width of the Gaussian function to use as kernel. Same width for all datapoints.
        How to estimate a fixed `fwhm` valid for all datapoints? If want a resolving power of approximately R in all datapoints, the `fwhm` can be computed by doing:
            fwhm = np.mean(x) / R
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dwindow : int
        Number of fwhms to define the half-window of data to use as the kernel size. I.e. when computing the convolution of the datapoint x_i, the Gaussian applied as kernel will be defined from `x_i - dwindow*fwhm` until `x_i + dwindow*fwhm`.
    """
    # Select width of the Gaussian
    if (resolution is None) and (fwhm is None):
        sys.exit('Must specify either `resolution` or `fwhm`.')
    if resolution is None:
        # Check fwhm is a valid number
        if isinstance(fwhm, (int, float)): 
            if verb: print('Use the same fwhm for each datapoint: {}'.format(fwhm))
        else: sys.exit('fwhm not valid {}'.format(fwhm))
    else:
        if verb: print('Use resolution: {} (different fwhm each datapoint)'.format(resolution))
        # Compute fwhm for each datapoint
        fwhm = x / resolution
        #sigma_new = fwhm / (2 * np.sqrt(2 * np.log(2)))
        #sigma = sigma_new
    return conv_gauss_custom(x, y, fwhm, dwindow=dwindow)

###############################################################################


# Save/Read spectrum

def spec_save_pkl_matrix(w, f, filout, verb=False):
    """
    Save pickle
    """
    tpl = {'w': w, 'f': f}

    with open(filout, 'wb') as handle:
        pickle.dump(tpl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

    if verb: print('FITS with processed template saved in {}'.format(filout))
    return


def spec_read_pkl_matrix(filin):
    """
    Read pickle
    """
    with open(filin, 'rb') as handle:
        a = pickle.load(handle)
    w = a['w']
    f = a['f']
    return w, f
