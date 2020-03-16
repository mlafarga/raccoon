#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from astropy.io import fits
import numpy as np

from . import fitsutils


# Number of orders
dicnord = {
    'HARPS': 72,
    'HARPN': 69,
}

###############################################################################


# FITS Header
# -----------

def headerkwinst(inst, outfail=np.nan):
    """Get the corresponding start of the header keywords depending on the instrument:
    - for HARPS, the keywords start with 'HIERARCH ESO...'
    - for HARPS-N, the keywords start with 'HIERARCH TNG...'
    If fail, return `outfail`, which by default is NaN.

    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Returns
    -------
    kwinst : {'HIERARCH TNG ', 'HIERARCH ESO '}
    """

    # Make sure `inst` is a string
    if not isinstance(inst, str):
        sys.exit('Instrument {} is a {}, not a string'.format(inst, type(inst)))

    # Get correct header keyword
    try:
        kwinst = 'HIERARCH ' + {'harps': 'ESO', 'harpn': 'TNG'}[inst.lower()] + ' '
    except:
        kwinst = outfail
        print('Not correct instrument: {}'.format(inst))

    return kwinst


def drs_e2ds_inst(filin, outfail=np.nan, ext=0):
    """Try to get the instrument (HARPS or HARPS-N) from the header keywords. If fail, return `outfail`, which by default is NaN.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before)

    Returns
    -------
    inst : {'harpn', 'harps'}
        Instrument name. Returns `outfail` (default NaN) if failed to find instrument in header keywords (may indicate wrong file).
    """

    # Read header
    if isinstance(filin, str): header = fitsutils.read_header(filin, ext=ext)
    elif isinstance(filin, fits.header.Header): header = filin

    # Identify instrument
    if 'ESO' in header['TELESCOP']: inst = 'harps'
    elif 'TNG' in header['TELESCOP']: inst = 'harpn'
    elif 'HARPS' in header['INSTRUME']: inst = 'harps'
    elif 'HARPN' in header['INSTRUME']: inst = 'harpn'
    else: inst = outfail

    return inst


def drs_e2ds_int_kwinst(filin, outfail=np.nan, ext=0):
    """Get the instrument name ('harpn' or 'harps') and the corresponding header keyword start ('HIERARCH TNG ' or 'HIERARCH ESO ').

    Uses `drs_e2ds_inst` to get the instrument and `headerkwinst` to get the header keyword start.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before)

    Returns
    -------
    inst : {'harpn', 'harps', np.nan}
        Instrument name. Returns NaN if failed to find instrument in header keywords (may indicate wrong file).
    kwinst : {'TNG', 'ESO'}
    """
    inst = drs_e2ds_inst(filin, outfail=outfail, ext=ext)
    kwinst = headerkwinst(inst, outfail=outfail)
    return inst, kwinst


# -----------------------------------------------------------------------------


# FITS data
# ---------

# Flux

def drs_fitsred_read(filin, ext=0):
    """
    Read an e2ds or a blaze reduced spectrum FITS (i.e. spectrum flux).
    """
    # Check if file exists
    if not os.path.isfile(filin): sys.exit('File does not exist: {}'.format(filin))

    # Read FITS
    with fits.open(filin) as hdulist:
        header = hdulist[ext].header
        f = hdulist[ext].data

    return f, header


# Wavelength from header keywords

def wpolycoeff(filin, inst=None, ext=0):
    """From e2ds header get the polynomial coefficients necessary to transform from pixels to wavelength.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before)
    inst : {'harpn', 'harps'} or None (default)
    # inst : {'harpn', 'harps', 'ESO', 'TNG'} or None (default)
    """

    # Read header
    if isinstance(filin, str): header = fitsutils.read_header(filin, ext=ext)
    elif isinstance(filin, fits.header.Header): header = filin

    # Get instrument kw
    if inst is None:
        inst, kwinst = drs_e2ds_int_kwinst(header, outfail=np.nan, ext=ext)
    elif inst.lower() == 'harpn' or inst.lower() == 'harps':
        inst = inst.lower()
        kwinst = headerkwinst(inst, outfail=np.nan)
    else:
        sys.exit('No correct instrument: {}'.format(inst))

    # Number of orders
    nord = dicnord[inst.upper()]
    ords = np.arange(0, nord, 1)

    # Get polynomial coefficients (4 per order) from header
    #  Polynomial coefficient numbers, 4 per order
    polydeg = header[kwinst + 'DRS CAL TH DEG LL']
    coeffnum = [np.arange(0+o*(polydeg+1), (polydeg+1)+o*(polydeg+1), 1) for o in ords]
    #  Polynomial coefficient values, 4 per order
    coeff = [[header[kwinst + 'DRS CAL TH COEFF LL{:d}'.format(j)] for j in coeffnum[o]] for o in ords]

    return coeff


def pix2wave(x, coeff):
    """Convert pixel to wavelength using the coefficients from the e2ds header, for a single echelle order.

    Parameters
    ----------
    x : 1d array
        Pixels.
    """
    w = coeff[0] + coeff[1] * x + coeff[2] * x**2 + coeff[3] * x**3
    return w


def pix2wave_echelle(x, coeff):
    nord = len(coeff)
    ords = np.arange(0, nord, 1)
    w = np.array([pix2wave(x, coeff[o]) for o in ords])
    return w


def drs_e2dsred_readw(filin, inst=None, npix=4096, ext=0):
    coeff = wpolycoeff(filin, inst=inst, ext=ext)
    pix = np.arange(0, npix, 1)
    w = pix2wave_echelle(pix, coeff)
    return w


# Complete spectrum: w, f, blaze (optional) and headers

def drs_e2dsred_read(filin, readblaze=True, dirblaze=None, filblaze=None, inst=None, exte2ds=0, extb=0):
    """
    Read e2ds reduced spectrum flux and wavelength, and optionally the blaze.

    The wavelength data is obtained from the header keywords in `filin` using `drs_e2dsred_readw`.

    The blaze is obtained if `readblaze` is True.
    The directory containing the blaze files by default is the same as the directory where the e2ds file is (`filin`), but can be changed with `dirblaze`.
    The blaze file by default is obtained from the header of the e2ds file `filin`: 'HIERARCH TNG DRS BLAZE FILE', but can be changed with `filblaze`.

    Parameters
    ----------
    filin : str
        Reduced e2ds file.
    readblaze : bool, default True
        Whether to read the blaze or not. If False, the returned blaze `b` is an array full of ones of the same shape as the spectrum, and the header `header` is NaN.
    dirblaze : str, default None
        Directory where the blaze file is. If None (default), it is assumed that it is in the same directory as the spectrum `filin`.
    filblaze : str, default None
        Blaze file. Use if want to obtain the blaze from a file different than the one specified in the header keyword 'HIERARCH TNG DRS BLAZE FILE'.
    """

    # Read e2ds flux
    f, header = drs_fitsred_read(filin, ext=exte2ds)

    # Get wavelength from header
    nord = len(f)
    npix = len(f[0])
    w = drs_e2dsred_readw(filin, inst=inst, nord=nord, npix=npix, ext=exte2ds)

    # Read blaze
    if readblaze:
        kwinst = headerkwinst(inst, outfail=np.nan)

        if dirblaze is None: dirblaze = os.path.dirname(filin)
        if filblaze is None: filblaze = header[kwinst + 'DRS BLAZE FILE']
        b, headerb = drs_fitsred_read(os.path.join(dirblaze, filblaze), ext=extb)
    else:
        b = np.ones_like(f)
        headerb = np.nan

    return w, f, b, header, headerb


# BJD

def drs_bjd(filin, inst, notfound=np.nan, ext=0, outfmt='single', name='bjd'):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}
    outfmt : {'single', 'dict'}
    name : str, None

    Notes
    -----
    HIERARCH ESO DRS BJD = 2457671.53027316 / Barycentric Julian Day
    """

    # Get instrument kw
    kwinst = headerkwinst(inst.lower(), outfail=np.nan)

    kw = kwinst + 'DRS BJD'
    if name is not None: name = {kw: name}
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext, names=name)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_bjd_lisobs(lisobs, inst, notfound=np.nan, ext=0, name='bjd'):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Return
    ------
    data : pandas dataframe
    """
    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = kwinst + 'DRS BJD'
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


# SNR

def drs_snr(filin, ords=None, notfound=np.nan, ext=0):
    """Get the SNR of the orders in `ords` (e.g. from keyword 'HIERARCH ESO DRS SPE EXT SN55').

    Work for both HARPS and HARPN observations.

    Uses:
        - `fitsutils.read_header`

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    ords : list of int, default None
        Orders from which to get the SNR. If None (default), get SNR for all the orders.
    notfound :
        Value to return if keyword not found in FITS header.
    ext : int, default 0
        Extension of the FITS from which to get the header. Only used if `filin` is a FITS file.

    Returns
    -------
    dic : dict
        Dictionary with the header keywords and their values.
    """
    pattern = '*DRS SPE EXT SN*'
    # Read header
    if isinstance(filin, str): header = fitsutils.read_header(filin, ext=ext)
    elif isinstance(filin, fits.header.Header): header = filin
    # Get SNR for all orders
    kws = header[pattern]  # astropy.io.fits.header.Header object

    # All orders
    if ords is None:
        vals = list(kws.values())
        kws = list(kws.keys())
        dic = dict(zip(kws, vals))  # dict object
    # Select orders
    else:
        # Make sure ords is a list-type of ints
        if not isinstance(ords, (list, tuple, np.ndarray)): ords = [ords]
        if not isinstance(ords[0], int): ords = [int(o) for o in ords]
        # Select orders
        # `int(re.search(r'\d+$', k).group())` used to find integers at the end of the string `k`
        dic = {k: v for k, v in kws.items() if int(re.search(r'\d+$', k).group()) in ords}

    return dic


def drs_snr_lisobs(lisobs, ords=None, notfound=np.nan, ext=0):
    """Get the SNR of the orders in `ords` (e.g. from keyword ''HIERARCH ESO DRS SPE EXT SN55') for the observations in `lisobs`.

    Uses:
        - `fitsutils.read_header`
        - `fitsutils.read_header_keywords_lisobs`

    Returns
    -------
    data : pandas dataframe
    """
    pattern = '*DRS SPE EXT SN*'

    # Get keywords: Read header of 1st observation
    filin = lisobs[0]
    if isinstance(filin, str): header = fitsutils.read_header(filin, ext=ext)
    elif isinstance(filin, fits.header.Header): header = filin
    # Get SNR for all orders
    kws = header[pattern]  # astropy.io.fits.header.Header object

    # All orders
    if ords is None:
        kws = list(kws.keys())
    # Select orders
    else:
        # Make sure ords is a list-type of ints
        if not isinstance(ords, (list, tuple, np.ndarray)): ords = [ords]
        if not isinstance(ords[0], int): ords = [int(o) for o in ords]
        # Select orders
        # `int(re.search(r'\d+$', k).group())` used to find integers at the end of the string `k`
        kws = [k for k in kws.keys() if int(re.search(r'\d+$', k).group()) in ords]

    # Check that HIERARCH in keywords
    # Needed because if search header using pattern the HIERARCH word disappears
    if not kws[0].startswith('HIERARCH'):
        kws = ['HIERARCH ' + i if not i.startswith('HIERARCH') else i for i in kws]

    # Read headers
    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext)
    return data


# RON

def drs_ron(filin, inst, notfound=np.nan, ext=0, outfmt='single'):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}
    outfmt : {'single', 'dict'}

    Notes
    -----
    HIERARCH ESO DRS CCD SIGDET = 4.69431553672547 / CCD Readout Noise [e-]
    HIERARCH ESO DRS CCD CONAD = 1.36 / CCD conv factor [e-/ADU]
    """

    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = kwinst + 'DRS CCD SIGDET'
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_ron_lisobs(lisobs, inst, notfound=np.nan, ext=0):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Return
    ------
    data : pandas dataframe
    """
    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = kwinst + 'DRS CCD SIGDET'
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext)
    return data


# Exposure time

def drs_exptime(filin, notfound=np.nan, ext=0, outfmt='single'):
    """
    """

    # # Get instrument kw
    # try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    # except: sys.exit('No correct instrument: {}'.format(inst))

    kw = 'EXPTIME'
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_exptime_lisobs(lisobs, notfound=np.nan, ext=0):
    kw = 'EXPTIME'
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext)
    return data


def drs_airmass_lisobs(lisobs, notfound=np.nan, ext=0):
    kw = 'AIRMASS'
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext)
    return data


# RV corrections

def drs_berv(filin, inst, notfound=np.nan, ext=0, outfmt='single', units='is'):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}
    outfmt : {'single', 'dict'}

    Notes
    -----
    HIERARCH ESO DRS BERV = -29.8640539483386 / Barycentric Earth Radial Velocity
    """

    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = kwinst + 'DRS BERV'
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)
    if units == 'is':
        dic[kw] = dic[kw] * 1.e3
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_berv_lisobs(lisobs, inst, notfound=np.nan, ext=0, units='is'):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Return
    ------
    data : pandas dataframe
    """
    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = kwinst + 'DRS BERV'
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, units=units)
    return data


def drs_bervdrift(filin, inst, notfound=np.nan, ext=0, outfmt='single'):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}
    outfmt : {'single', 'dict'}
    """

    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = kwinst + 'DRS DRIFT CCF RV'
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_drift_lisobs(lisobs, inst, notfound=np.nan, ext=0):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Return
    ------
    data : pandas dataframe
    """
    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = kwinst + 'DRS DRIFT CCF RV'
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext)
    return data


def drs_rvcorrection(filin, inst, notfound=np.nan, ext=0):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Returns
    -------
    dic : dict
        Dictionary with the header keywords and their values. All units in IS.

    Notes
    -----
    HIERARCH ESO DRS BERV = -29.8640539483386 / Barycentric Earth Radial Velocity
    HIERARCH ESO DRS DRIFT CCF RV = 0. / CCF RV Drift [m/s]
    """

    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = [kwinst + 'DRS BERV', kwinst + 'DRS DRIFT CCF RV']
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)

    # m/s
    dic[kwinst + 'DRS BERV'] = dic[kwinst + 'DRS BERV'] * 1.e3

    # Compute RV correction
    shift = - dic[kwinst + 'DRS BERV'] + dic[kwinst + 'DRS DRIFT CCF RV']
    shifterr = np.nan

    return shift, shifterr, dic


def drs_rvcorrection_lisobs(lisobs, inst, name='shift', notfound=np.nan, ext=0):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Return
    ------
    data : pandas dataframe
    """
    # Get instrument kw
    try: kwinst = headerkwinst(inst.lower(), outfail=np.nan)
    except: sys.exit('No correct instrument: {}'.format(inst))

    kw = [kwinst + 'DRS BERV', kwinst + 'DRS DRIFT CCF RV']
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext)

    # m/s
    data[kwinst + 'DRS BERV'] = data[kwinst + 'DRS BERV'] * 1.e3

    # Compute RV correction
    shift = - data[kwinst + 'DRS BERV'] + data[kwinst + 'DRS DRIFT CCF RV']
    shifterr = np.nan
    data['shift'] = shift
    data['shifterr'] = shifterr

    # Change dataframe keywords
    data.rename(columns={kwinst + 'DRS BERV': 'berv', kwinst + 'DRS DRIFT CCF RV': 'drift'}, inplace=True)

    return shift, shifterr, data
