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

# Read header

def drs_fitsred_header(filin, verbose=True):
    """
    Read an echelle reduced spectrum FITS header.
    Should work for all `_S2D_A.fits`, `_S2D_SKYSUB_A.fits`, `_S2D_BLAZE_A.fits`, 

    Parameters
    ----------
    filin : str
        FITS file.

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
    if verbose: print('Reading header from', filin)
    with fits.open(filin) as hdulist:
        # Header
        header = hdulist[0].header

    return header


def drs_fitsred_header_lisobs(lisfilin, verbose=True):
    lisheader = []
    nobs = len(lisfilin)
    for i, filin in enumerate(lisfilin):
        if verbose: print('{}/{} Reading header from {}'.format(i+1, nobs, filin))
        lisheader.append(drs_fitsred_header(filin, verbose=False))
    return lisheader


# Read all reduced data

def drs_fitsred_read(filin, qualdata2mask=True, w='vac'):
    """
    Read an echelle reduced spectrum FITS.
    Should work for all `_S2D_A.fits`, `_S2D_SKYSUB_A.fits`, `_S2D_BLAZE_A.fits`, 

    Parameters
    ----------
    filin
    qualdata2mask : bool, default: True

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


# Telescope number, need for certain keywords

def headertelescopenumber(filin, outfail=np.nan, ext=0):
    """Get the corresponding start of the header keywords depending on the unit telescope used: keywords can be 'HIERARCH ESO TEL1 something', 'HIERARCH ESO TEL2 something', 'HIERARCH ESO TEL3 something', or 'HIERARCH ESO TEL4 something' since there are 4 telescopes.
    If fail, return `outfail`, which by default is NaN.

    Returns
    -------
    kwinst : {'HIERARCH TNG ', 'HIERARCH ESO '}

    Notes
    -----
    TELESCOP= 'ESO-VLT-U1'         / ESO <TEL>
    HIERARCH ESO OCS TEL NO =    1 / Number of active telescopes (1 - 4).
    HIERARCH ESO OCS TEL1 ST =   T / Availability of the telescope.
    HIERARCH ESO OCS TEL2 ST =   F / Availability of the telescope.
    HIERARCH ESO OCS TEL3 ST =   F / Availability of the telescope.
    HIERARCH ESO OCS TEL4 ST =   F / Availability of the telescope.
    """

    # Read header
    if isinstance(filin, str): header = fitsutils.read_header(filin, ext=ext)
    elif isinstance(filin, fits.header.Header): header = filin

    # Identify telescope number
    try:
        telnumber = header['TELESCOP'].replace('ESO-VLT-U', '')
    except:
        telnumber = outfail
        print('Telescope number not found: {}'.format(telnumber))

    return telnumber


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


# Airmass

def drs_airmass_start_lisobs(lisobs, notfound=np.nan, ext=0, name='airmassstart'):
    """
    HIERARCH ESO TEL1 AIRM END = 2.551 / Airmass at end
    HIERARCH ESO TEL1 AIRM START = 2.612 / Airmass at start
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} AIRM START'.format(telnumber)

    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_airmass_end_lisobs(lisobs, notfound=np.nan, ext=0, name='airmassend'):
    """
    HIERARCH ESO TEL1 AIRM END = 2.551 / Airmass at end
    HIERARCH ESO TEL1 AIRM START = 2.612 / Airmass at start
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} AIRM END'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


# ADC stuff

def drs_adc2ra_lisobs(lisobs, notfound=np.nan, ext=0, name='adc2ra'):
    """
    """
    telnumber = headertelescopenumber(lisobs[0])
    if telnumber == '1': telnumber = ''
    kw = 'HIERARCH ESO INS{} ADC2 RA'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_adc2sens1_lisobs(lisobs, notfound=np.nan, ext=0, name='adc2sens1'):
    """
    """
    telnumber = headertelescopenumber(lisobs[0])
    if telnumber == '1': telnumber = ''
    kw = 'HIERARCH ESO INS{} ADC2 SENS1'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_adc2temp_lisobs(lisobs, notfound=np.nan, ext=0, name='adc2temp'):
    """
    """
    telnumber = headertelescopenumber(lisobs[0])
    if telnumber == '1': telnumber = ''
    kw = 'HIERARCH ESO INS{} ADC2 TEMP'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_fluxcorrcheck_lisobs(lisobs, notfound=np.nan, ext=0, name='fluxcorrcheck'):
    """
    """
    kw = 'HIERARCH ESO QC SCIRED FLUX CORR CHECK'
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_domestatus_lisobs(lisobs, notfound=np.nan, ext=0, name='domestatus'):
    """
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} DOME STATUS'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


# Seeing

"""
From https://www.eso.org/rm/api/v1/public/releaseDescriptions/176
    The fourth flag is related to the seeing during the observation. The pipeline corrects the extracted
flux for losses due to the fibre entrance (“slit loss”) which depends on the seeing. The seeing value
is recorded in the “HIERARCH ESO TEL<N> IA FWHM” header keywords. In exceptional cases, the
keyword contains an unrealistically low or high value which leads to an unreliable flux correction.
Flag #4 is set in such a case.

IA detector: Image Analyis detector

AMBI FWHM: Astronomical Site Monitor seeing

TEL.IA.FWHMLIN Delivered seeing on Image Analysis detector at ∼550-nm.
TEL.IA.FWHMLINOBS Delivered seeing on Image Analysis detector at ∼550-nm.

"""

def drs_seeingambi_start_lisobs(lisobs, notfound=np.nan, ext=0, name='seeingambistart'):
    """Observatory seeing
    HIERARCH ESO TEL1 AMBI FWHM END = 0.51 / [arcsec] Observatory Seeing queried fro
    HIERARCH ESO TEL1 AMBI FWHM START = 0.49 / [arcsec] Observatory Seeing queried f
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} AMBI FWHM START'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_seeingambi_end_lisobs(lisobs, notfound=np.nan, ext=0, name='seeingambiend'):
    """Observatory seeing
    HIERARCH ESO TEL1 AMBI FWHM END = 0.51 / [arcsec] Observatory Seeing queried fro
    HIERARCH ESO TEL1 AMBI FWHM START = 0.49 / [arcsec] Observatory Seeing queried f
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} AMBI FWHM END'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_seeingia_airmasscorrected_lisobs(lisobs, notfound=np.nan, ext=0, name='seeingiaairmasscorrected'):
    """
    HIERARCH ESO TEL1 IA FWHM = 0.71 / [arcsec] Delivered seeing corrected by airmas
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} IA FWHM'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_seeingia_linear_lisobs(lisobs, notfound=np.nan, ext=0, name='seeingialinear'):
    """
    HIERARCH ESO TEL1 IA FWHMLIN = 0.85 / Delivered seeing on IA detector (linear fi
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} IA FWHMLIN'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_seeingia_linearobs_lisobs(lisobs, notfound=np.nan, ext=0, name='seeingialinearobs'):
    """
    HIERARCH ESO TEL1 IA FWHMLINOBS = 1.5 / Delivered seeing on IA detector (linear 
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} IA FWHMLINOBS'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


# Integrated water vapor
"""
TEL AMBI.IWV.START Average of the Integrated Water Vapor measurements towards
zenith over the previous 2min at start of exposure. Implemented soon.
"""
def drs_iwv_start_lisobs(lisobs, notfound=np.nan, ext=0, name='iwvstart'):
    """
    HIERARCH ESO TEL1 AMBI IWV END = 12.12 / Integrated Water Vapor                 
    HIERARCH ESO TEL1 AMBI IWV START = 12.12 / Integrated Water Vapor               
    HIERARCH ESO TEL1 AMBI IWV30D END = 12.12 / IWV at 30deg elev.                  
    HIERARCH ESO TEL1 AMBI IWV30D START = 11.87 / IWV at 30deg elev.                
    HIERARCH ESO TEL1 AMBI IWV30DSTD END = 0.08 / IWV at 30deg elev.                
    HIERARCH ESO TEL1 AMBI IWV30DSTD START = 0.18 / IWV at 30deg elev.  
    HIERARCH ESO TEL1 AMBI IWVSTD END = 0. / Standard Deviation of Integrated Wate  
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} AMBI IWV START'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_iwv_end_lisobs(lisobs, notfound=np.nan, ext=0, name='iwvend'):
    """ 
    HIERARCH ESO TEL1 AMBI IWV END = 12.12 / Integrated Water Vapor                 
    HIERARCH ESO TEL1 AMBI IWV START = 12.12 / Integrated Water Vapor               
    HIERARCH ESO TEL1 AMBI IWV30D END = 12.12 / IWV at 30deg elev.                  
    HIERARCH ESO TEL1 AMBI IWV30D START = 11.87 / IWV at 30deg elev.                
    HIERARCH ESO TEL1 AMBI IWV30DSTD END = 0.08 / IWV at 30deg elev.                
    HIERARCH ESO TEL1 AMBI IWV30DSTD START = 0.18 / IWV at 30deg elev.  
    HIERARCH ESO TEL1 AMBI IWVSTD START = 0. / Standard Deviation of Integrated Wa 
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} AMBI IWV END'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data


def drs_ambirhum_lisobs(lisobs, notfound=np.nan, ext=0, name='ambirhum'):
    """ 
    HIERARCH ESO TEL1 AMBI RHUM = 38.5 / [%] Observatory ambient relative humidity q
    `TEL` can change from 1 to 4! Assume all observations taken with the same telescope
    """
    telnumber = headertelescopenumber(lisobs[0])
    kw = 'HIERARCH ESO TEL{} AMBI RHUM'.format(telnumber)
    if name is not None: name = {kw: name}
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=name)
    return data
    
# Other keywords to consider
# HIERARCH ESO TEL1 AMBI PRES END = 744.2 / [hPa] Observatory ambient air pressur 
# HIERARCH ESO TEL1 AMBI PRES START = 744.2 / [hPa] Observatory ambient air press 
# HIERARCH ESO TEL1 AMBI WINDDIR = 76.5 / [deg] Observatory ambient wind direction
# HIERARCH ESO TEL1 AMBI WINDSP = 6.53 / [m/s] Observatory ambient wind speed quer



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
    # names = {'HIERARCH ESO QC CCF RV': 'ccfrv', 'HIERARCH ESO QC CCF RV ERROR': 'ccfrverr', 'HIERARCH ESO QC CCF FWHM': 'ccffwhm', 'HIERARCH ESO QC CCF FWHM ERROR': 'ccffwhmerr', 'HIERARCH ESO QC CCF CONTRAST': 'ccfcontrast', 'HIERARCH ESO QC CCF CONTRAST ERROR': 'ccfcontrasterr', 'HIERARCH ESO QC CCF CONTINUUM': 'ccfcont', 'HIERARCH ESO QC CCF MASK': 'ccfmask', 'HIERARCH ESO QC CCF FLUX ASYMMETRY': 'ccffasy', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR': 'ccffasyerr', 'HIERARCH ESO QC CCF BIS SPAN': 'ccfbis', 'HIERARCH ESO QC CCF BIS SPAN ERROR': 'ccfbiserr', 'HIERARCH PIPELINE MASK TIMESTAMP': 'ccfmasktimestamp'}
    names = {'HIERARCH ESO QC CCF RV': 'ccfrv', 'HIERARCH ESO QC CCF RV ERROR': 'ccfrverr', 'HIERARCH ESO QC CCF FWHM': 'ccffwhm', 'HIERARCH ESO QC CCF FWHM ERROR': 'ccffwhmerr', 'HIERARCH ESO QC CCF CONTRAST': 'ccfcontrast', 'HIERARCH ESO QC CCF CONTRAST ERROR': 'ccfcontrasterr', 'HIERARCH ESO QC CCF CONTINUUM': 'ccfcont', 'HIERARCH ESO QC CCF MASK': 'ccfmask', 'HIERARCH ESO QC CCF FLUX ASYMMETRY': 'ccffasy', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR': 'ccffasyerr', 'HIERARCH ESO QC CCF BIS SPAN': 'ccfbis', 'HIERARCH ESO QC CCF BIS SPAN ERROR': 'ccfbiserr', 'HIERARCH PIPELINE MASK TIMESTAMP': 'ccfmasktimestamp'}

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
    names = {'HIERARCH ESO QC CCF RV': 'ccfrv', 'HIERARCH ESO QC CCF RV ERROR': 'ccfrverr', 'HIERARCH ESO QC CCF FWHM': 'ccffwhm', 'HIERARCH ESO QC CCF FWHM ERROR': 'ccffwhmerr', 'HIERARCH ESO QC CCF CONTRAST': 'ccfcontrast', 'HIERARCH ESO QC CCF CONTRAST ERROR': 'ccfcontrasterr', 'HIERARCH ESO QC CCF CONTINUUM': 'ccfcont', 'HIERARCH ESO QC CCF MASK': 'ccfmask', 'HIERARCH ESO QC CCF FLUX ASYMMETRY': 'ccffasy', 'HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR': 'ccffasyerr', 'HIERARCH ESO QC CCF BIS SPAN': 'ccfbis', 'HIERARCH ESO QC CCF BIS SPAN ERROR': 'ccfbiserr', 'HIERARCH PIPELINE MASK TIMESTAMP': 'ccfmasktimestamp'}

    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext, names=names)
    return data


# # CCF

# def drs_ccf_read(filin):
#     with fits.open(filin) as hdulist:
#         header = hdulist[0].header
#         lisccf = hdulist[0].data
#     return header, lisccf