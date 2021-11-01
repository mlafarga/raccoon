#!/usr/bin/env python
"""
ESPRESSO utils

Notes
-----


_S2D_A.fits structure
---------------------

Filename: r.ESPRE.2021-01-01T02:41:31.734_S2D_A.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU    1480   ()      
  1  SCIDATA       1 ImageHDU        10   (9111, 170)   float32   
  2  ERRDATA       1 ImageHDU        10   (9111, 170)   float32   
  3  QUALDATA      1 ImageHDU        12   (9111, 170)   int16 (rescales to uint16)   
  4  WAVEDATA_VAC_BARY    1 ImageHDU        10   (9111, 170)   float64   
  5  WAVEDATA_AIR_BARY    1 ImageHDU        10   (9111, 170)   float64   
  6  DLLDATA_VAC_BARY    1 ImageHDU        10   (9111, 170)   float64   
  7  DLLDATA_AIR_BARY    1 ImageHDU        10   (9111, 170)   float64 
"""
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from astropy.io import fits
import numpy as np

from . import fitsutils


# Number of orders
nord = 170

# Number of pixels
npix = 9111

# Header keywords stars
kwinst = 'HIERARCH ESO'


###############################################################################


# FITS Header
# -----------



# -----------------------------------------------------------------------------


# FITS data
# ---------

# Read all reduced data

def drs_fitsred_read(filin, qualdata2mask=True, w='vac'):
    """
    Read an echelle reduced spectrum FITS.
    Should work for all `_S2D_A.fits`, `_S2D_SKYSUB_A.fits`, `_S2D_BLAZE_A.fits`, 

    Parameters
    ----------
    filin
    qualdata2mask : bool, default: True
        return 

    Notes
    -----
    - Wavelengths are BERV corrected and in air and vacuum (2 fits extensions)
    - `QUALDATA` seems to only remove pixels where flux is 0, not pixel with mostly noise or spikes.

    _S2D_A.fits structure
    ---------------------

    Filename: r.ESPRE.2021-01-01T02:41:31.734_S2D_A.fits
    
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU    1480   ()      
      1  SCIDATA       1 ImageHDU        10   (9111, 170)   float32   
      2  ERRDATA       1 ImageHDU        10   (9111, 170)   float32   
      3  QUALDATA      1 ImageHDU        12   (9111, 170)   int16 (rescales to uint16)   
      4  WAVEDATA_VAC_BARY    1 ImageHDU        10   (9111, 170)   float64   
      5  WAVEDATA_AIR_BARY    1 ImageHDU        10   (9111, 170)   float64   
      6  DLLDATA_VAC_BARY    1 ImageHDU        10   (9111, 170)   float64   
      7  DLLDATA_AIR_BARY    1 ImageHDU        10   (9111, 170)   float64 
    """
    # Check if file exists
    if not os.path.isfile(filin): sys.exit('File does not exist: {}'.format(filin))

    # Read FITS
    with fits.open(filin) as hdulist:
        # Header
        header = hdulist[0].header
        # Flux and error
        f = hdulist['SCIDATA'].data
        sf = hdulist['ERRDATA'].data
        # Quality pixel map
        q = hdulist['QUALDATA'].data
        #   Quality pixel map to boolean mask
        mq = ~np.array(hdulist['QUALDATA'].data, dtype=bool)
        # Wavelength
        w = hdulist['WAVEDATA_VAC_BARY'].data
        wair = hdulist['WAVEDATA_AIR_BARY'].data
        # Detector pixel sizes
        dll = hdulist['DLLDATA_VAC_BARY'].data
        dllair = hdulist['DLLDATA_AIR_BARY'].data

    return w, wair, f, sf, q, mq, dll, dllair, header


# BJD

def drs_bjd(filin, notfound=np.nan, ext=0, outfmt='single', name='bjd'):
    """
    Parameters
    ----------
    outfmt : {'single', 'dict'}
    name : str, None

    Notes
    -----
    HIERARCH ESO QC BJD = 2459215.61634838 / Barycentric Julian date (TDB) [JD]
    """

    kw = 'HIERARCH ESO QC BJD'
    if name is not None: name = {kw: name}
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext, names=name)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_bjd_lisobs(lisobs, notfound=np.nan, ext=0, name='bjd'):
    """
    Parameters
    ----------

    Return
    ------
    data : pandas dataframe
    """

    kw = 'HIERARCH ESO QC BJD'
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


# SNR

def drs_snr(filin, ords=None, notfound=np.nan, ext=0, name='snro'):
    """Get the SNR of the orders in `ords` (e.g. from keyword 'HIERARCH ESO QC ORDER10 SNR').

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
    name : str
        Change keywords to `name` plus the corresponding order (from 0 to nord, so will subtract -1 to the orders listed in the header)

    Returns
    -------
    dic : dict
        Dictionary with the header keywords and their values.
    """

    pattern = 'ESO QC ORDER* SNR'
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

    # Change keywords names
    if name is not None:
        dicnew = {}
        for kw, v in dic.items():
            kwnew = kw.replace('ESO QC ORDER', '')
            kwnew = kwnew.replace('SNR', '')
            kwnew = kwnew.strip()
            kwnew = str(int(kwnew) - 1)
            dicnew[name + kwnew] = v
        dic = dicnew

    return dic


def drs_snr_lisobs(lisobs, ords=None, notfound=np.nan, ext=0, name='snro'):
    """Get the SNR of the orders in `ords` (e.g. from keyword 'HIERARCH ESO QC ORDER10 SNR') for the observations in `lisobs`.

    Uses:
        - `fitsutils.read_header`
        - `fitsutils.read_header_keywords_lisobs`

    Returns
    -------
    data : pandas dataframe
    """
    pattern = 'ESO QC ORDER* SNR'

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

    if name is not None:
        columns = {kw: name+str(o) for o, kw in enumerate(data.columns)}
        data.rename(columns=columns, inplace=True)
    return data


# Exposure time

def drs_exptime(filin, notfound=np.nan, ext=0, name='exptime', outfmt='single'):
    """
    """
    kw = 'EXPTIME'
    if name is not None: name = {kw: name}
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext, names=name)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_exptime_lisobs(lisobs, notfound=np.nan, ext=0, name='exptime'):
    kw = 'EXPTIME'
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_airmass_start_lisobs(lisobs, notfound=np.nan, ext=0, name='airmassstart'):
    """
    HIERARCH ESO TEL1 AIRM END = 2.551 / Airmass at end
    HIERARCH ESO TEL1 AIRM START = 2.612 / Airmass at start
    `TEL` can change from 1 to 4! 
    """
    kw = 'HIERARCH ESO TEL1 AIRM START'
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_airmass_end_lisobs(lisobs, notfound=np.nan, ext=0, name='airmassend'):
    """
    HIERARCH ESO TEL1 AIRM END = 2.551 / Airmass at end
    HIERARCH ESO TEL1 AIRM START = 2.612 / Airmass at start
    """
    kw = 'HIERARCH ESO TEL1 AIRM END'
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


# RV corrections

def drs_berv(filin, notfound=np.nan, ext=0, name='berv', outfmt='single', units='is'):
    """
    Parameters
    ----------
    outfmt : {'single', 'dict'}

    Notes
    -----
    HIERARCH ESO QC BERV = 21.1484287835869 / Barycentric correction [km/s]
    """

    kw = 'HIERARCH ESO QC BERV'
    if name is not None: name = {kw: name}
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext, names=name)
    if units == 'is':
        dic[kw] = dic[kw] * 1.e3  # m/s
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def drs_berv_lisobs(lisobs, notfound=np.nan, ext=0, name='berv', units='is'):
    """
    Parameters
    ----------
    inst : {'harpn', 'harps'}

    Return
    ------
    data : pandas dataframe
    """
    kw = 'HIERARCH ESO QC BERV'
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    if units == 'is':
        data = data * 1.e3  # m/s
    return data


def drs_ccfkw(filin, notfound=np.nan, ext=0):
    """
    Notes
    -----
    HIERARCH ESO QC CCF RV = 23.5252079094442 / Radial velocity [km/s]
    HIERARCH ESO QC CCF RV ERROR = 0.00232336354823161 / Uncertainty on radial veloc
    HIERARCH ESO QC CCF FWHM = 10.2681818852801 / CCF FWHM [km/s]
    HIERARCH ESO QC CCF FWHM ERROR = 0.00464672709646322 / Uncertainty on CCF FWHM [
    HIERARCH ESO QC CCF CONTRAST = 53.1194717692313 / CCF contrast %
    HIERARCH ESO QC CCF CONTRAST ERROR = 0.0240384998607928 / CCF contrast error %
    HIERARCH ESO QC CCF CONTINUUM = 234054.424077978 / CCF continuum level [e-]
    HIERARCH ESO QC CCF MASK = 'F9      ' / CCF mask used
    HIERARCH ESO QC CCF FLUX ASYMMETRY = 0.0326534615419152 / CCF asymmetry (km/s)
    HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR = 0.00334680283931161 / CCF asymmetry e
    HIERARCH ESO QC CCF BIS SPAN = -0.0443801106731208 / CCF bisector span (km/s)
    HIERARCH ESO QC CCF BIS SPAN ERROR = 0.00464672709646322 / CCF bisector span err
    HIERARCH PIPELINE MASK TIMESTAMP = '2021-08-17T02:23:05' / Header Mask generatio
    """
    kws = ['HIERARCH ESO QC CCF RV', 'HIERARCH ESO QC CCF RV ERROR', 'HIERARCH ESO QC CCF FWHM', 'HIERARCH ESO QC CCF FWHM ERROR', 'HIERARCH ESO QC CCF CONTRAST', 'HIERARCH ESO QC CCF CONTRAST ERROR', 'HIERARCH ESO QC CCF CONTINUUM', 'HIERARCH ESO QC CCF MASK', 'HIERARCH ESO QC CCF FLUX ASYMMETRY', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR', 'HIERARCH ESO QC CCF BIS SPAN', 'HIERARCH ESO QC CCF BIS SPAN ERROR', 'HIERARCH PIPELINE MASK TIMESTAMP']
    # names = {'HIERARCH ESO QC CCF RV': 'ccfrv', 'HIERARCH ESO QC CCF RV ERROR': 'ccfrverr', 'HIERARCH ESO QC CCF FWHM': 'ccffwhm', 'HIERARCH ESO QC CCF FWHM ERROR': 'ccffwhmerr', 'HIERARCH ESO QC CCF CONTRAST': 'ccfcontrast', 'HIERARCH ESO QC CCF CONTRAST ERROR': 'ccfconstrasterr', 'HIERARCH ESO QC CCF CONTINUUM': 'ccfcont', 'HIERARCH ESO QC CCF MASK': 'ccfmask', 'HIERARCH ESO QC CCF FLUX ASYMMETRY': 'ccffasy', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR': 'ccffasyerr', 'HIERARCH ESO QC CCF BIS SPAN': 'ccfbis', 'HIERARCH ESO QC CCF BIS SPAN ERROR': 'ccfbiserr', 'HIERARCH PIPELINE MASK TIMESTAMP': 'ccfmasktimestamp'}
    names = {'HIERARCH ESO QC CCF RV': 'ccfrv', 'HIERARCH ESO QC CCF RV ERROR': 'ccfrverr', 'HIERARCH ESO QC CCF FWHM': 'ccffwhm', 'HIERARCH ESO QC CCF FWHM ERROR': 'ccffwhmerr', 'HIERARCH ESO QC CCF CONTRAST': 'ccfcontrast', 'HIERARCH ESO QC CCF CONTRAST ERROR': 'ccfconstrasterr', 'HIERARCH ESO QC CCF CONTINUUM': 'ccfcont', 'HIERARCH ESO QC CCF MASK': 'ccfmask', 'HIERARCH ESO QC CCF FLUX ASYMMETRY': 'ccffasy', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR': 'ccffasyerr', 'HIERARCH ESO QC CCF BIS SPAN': 'ccfbis', 'HIERARCH ESO QC CCF BIS SPAN ERROR': 'ccfbiserr', 'HIERARCH PIPELINE MASK TIMESTAMP': 'ccfmasktimestamp'}

    data = fitsutils.read_header_keywords(filin, kws, notfound=notfound, ext=ext, names=names)
    return data


def drs_ccfkw_lisobs(lisobs, notfound=np.nan, ext=0):
    """
    Notes
    -----
    HIERARCH ESO QC CCF RV = 23.5252079094442 / Radial velocity [km/s]
    HIERARCH ESO QC CCF RV ERROR = 0.00232336354823161 / Uncertainty on radial veloc
    HIERARCH ESO QC CCF FWHM = 10.2681818852801 / CCF FWHM [km/s]
    HIERARCH ESO QC CCF FWHM ERROR = 0.00464672709646322 / Uncertainty on CCF FWHM [
    HIERARCH ESO QC CCF CONTRAST = 53.1194717692313 / CCF contrast %
    HIERARCH ESO QC CCF CONTRAST ERROR = 0.0240384998607928 / CCF contrast error %
    HIERARCH ESO QC CCF CONTINUUM = 234054.424077978 / CCF continuum level [e-]
    HIERARCH ESO QC CCF MASK = 'F9      ' / CCF mask used
    HIERARCH ESO QC CCF FLUX ASYMMETRY = 0.0326534615419152 / CCF asymmetry (km/s)
    HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR = 0.00334680283931161 / CCF asymmetry e
    HIERARCH ESO QC CCF BIS SPAN = -0.0443801106731208 / CCF bisector span (km/s)
    HIERARCH ESO QC CCF BIS SPAN ERROR = 0.00464672709646322 / CCF bisector span err
    HIERARCH PIPELINE MASK TIMESTAMP = '2021-08-17T02:23:05' / Header Mask generatio
    """
    kws = ['HIERARCH ESO QC CCF RV', 'HIERARCH ESO QC CCF RV ERROR', 'HIERARCH ESO QC CCF FWHM', 'HIERARCH ESO QC CCF FWHM ERROR', 'HIERARCH ESO QC CCF CONTRAST', 'HIERARCH ESO QC CCF CONTRAST ERROR', 'HIERARCH ESO QC CCF CONTINUUM', 'HIERARCH ESO QC CCF MASK', 'HIERARCH ESO QC CCF FLUX ASYMMETRY', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR', 'HIERARCH ESO QC CCF BIS SPAN', 'HIERARCH ESO QC CCF BIS SPAN ERROR', 'HIERARCH PIPELINE MASK TIMESTAMP']
    names = {'HIERARCH ESO QC CCF RV': 'ccfrv', 'HIERARCH ESO QC CCF RV ERROR': 'ccfrverr', 'HIERARCH ESO QC CCF FWHM': 'ccffwhm', 'HIERARCH ESO QC CCF FWHM ERROR': 'ccffwhmerr', 'HIERARCH ESO QC CCF CONTRAST': 'ccfcontrast', 'HIERARCH ESO QC CCF CONTRAST ERROR': 'ccfconstrasterr', 'HIERARCH ESO QC CCF CONTINUUM': 'ccfcont', 'HIERARCH ESO QC CCF MASK': 'ccfmask', 'HIERARCH ESO QC CCF FLUX ASYMMETRY': 'ccffasy', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR': 'ccffasyerr', 'HIERARCH ESO QC CCF BIS SPAN': 'ccfbis', 'HIERARCH ESO QC CCF BIS SPAN ERROR': 'ccfbiserr', 'HIERARCH PIPELINE MASK TIMESTAMP': 'ccfmasktimestamp'}

    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext, names=names)
    return data


# # CCF

# def drs_ccf_read(filin):
#     with fits.open(filin) as hdulist:
#         header = hdulist[0].header
#         lisccf = hdulist[0].data
#     return header, lisccf