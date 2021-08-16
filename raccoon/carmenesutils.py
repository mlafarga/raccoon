#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import glob
import os
import sys

from astropy.io import fits
import numpy as np
import pandas as pd

from . import fitsutils

###############################################################################


# -----------------------------------------------------------------------------
#
# CARACAL utils
#
# -----------------------------------------------------------------------------

# FITS
# ----

def caracal_fitsred_read(filin):
    """
    Read CARACAL reduced spectrum FITS.
    """
    # Read FITS
    with fits.open(filin) as hdulist:
        header = hdulist[0].header
        w = hdulist['wave'].data
        f = hdulist['spec'].data
        sf = hdulist['sig'].data
        c = hdulist['cont'].data
    return w, f, sf, c, header


def caracal_fitsrednir_divide_ords(**kwargs):
    """
    NIR discontinuity at the center of the order: between pixel 2039 and pixel 2040.
    """
    npix = len(kwargs['w'][0])
    pixcut = int(npix/2)

    nord = len(kwargs['w'])
    ords = np.arange(0, nord, 1)
    # nord_new = nord*2
    # ords_new = np.arange(0, nord_new, 1)

    kwargs_new = {}
    for k, v in kwargs.items():
        kwargs_new[k] = []
        for o in ords:
            kwargs_new[k].append(kwargs[k][o][:pixcut])
            kwargs_new[k].append(kwargs[k][o][pixcut:])
    for k, v in kwargs_new.items():
        kwargs_new[k] = np.asarray(v)
    return kwargs_new


# Observations
# ------------

def caracal_lisobs_obj(dirin, pattern='car*_A.fits'):
    """Get list of observations of a specific object.

    Parameters
    ----------
    dirin : str
        Directory containing the FITS files.
    pattern : str (default: `car*_A.fits`)
        Search pattern.

    Return
    ------
    lisobs : list
        List of the full path of the observations inside `dirin` following `pattern`.
    lisfilobs : list
        List of only the filename (so not the full path) of the observations inside `dirin` following `pattern`.
    """

    # Path + filename
    lisobs = np.sort(glob.glob(os.path.join(dirin, pattern)))

    # Filename
    lisfilobs = np.array([os.path.basename(obs) for obs in lisobs])

    return lisobs, lisfilobs


def caracal_lisobj(dirin, pattern='J*/'):
    """List all objects in `dirin` that follow `pattern`.

    Assume input directory has the following structure:
        dirin
        |-- obj1
        |-- obj2
        |-- ...
        |-- objk

    Parameters
    ----------
    dirin : str
        General CARACAL directory, containing the directories of all the objects.
    pattern : str (default: `J*`)
        Search pattern for the objects.

    Returns
    -------
    lisdirobj : list of str
        List with the full path of the object directories.
    lisobj : list of str
        List with only the object names (so not the full path).
    """

    # Complete path
    lisdirobj = np.sort(glob.glob(os.path.join(dirin, pattern)))

    # Object name
    lisobj = np.array([os.path.basename(os.path.normpath(dirobj)) for dirobj in lisdirobj])

    return lisdirobj, lisobj


def caracal_lisobs_allobj(dirin, lisobj=None, patternobj='J*/', patternobs='car*_A.fits'):
    """Get list of observations of all objects.

    Assume input directory has the following structure:
        dirin
        |-- obj1
        |   |-- obs1
        |   |-- obs2
        |   |-- ...
        |   |-- obsn
        |-- obj2
        |   |-- obs1
        |   |-- obs2
        |   |-- ...
        |   |-- obsm
        |-- ...
        |-- objk

    Parameters
    ----------
    dirin : str
        Directory containing the FITS files.
    patternobs : str (default: `car*_A.fits`)
        Search pattern for the observations.
    lisobj : list of str (default None)
        List with the names of the objects from which to get the observations.
        If None (default), all the directories inside `dirin` that follow `patternobj` are considered objects.
    patternobj : str (default: `J*`)
        Search pattern for the objects.

    Returns
    -------
    lisobs : dict of lists
        Dictionary: keys = objects, values = list with the full path of all the observations.
    lisfilobs : dict of lists
        Dictionary: keys = objects, values = list with the filename (so not the full path) of all the observations.
    """

    # Targets
    if lisobj is None:
        lisdirobj, lisobj = caracal_lisobj(dirin, pattern=patternobj)
        # # Complete path
        # lisdirobj = np.sort(glob.glob(os.path.join(dirin, patternobj)))
        # # Object name
        # lisobj = [os.path.basename(os.path.normpath(dirobj)) for dirobj in lisdirobj]

    # Observations per object
    lisobs, lisfilobs = {}, {}
    for obj in lisobj:
        lisobs[obj], lisfilobs[obj] = caracal_lisobs_obj(os.path.join(dirin, obj), pattern=patternobs)

    return lisobs, lisfilobs


# Lines
# -----

def read_stronglines_csv(filin='PhD/Data/CARMENES_GTO/caracal/stronglines.csv'):
    """
    """
    columns = ['id', 'name', 'w', 'str', 'ref', 'comment']
    data = pd.read_csv(filin, sep=',', header=None, names=columns, comment='#')
    return data


# Getting data from CARACAL FITS
# ------------------------------

def caracal_berv(filin, notfound=np.nan, ext=0, outfmt='single'):
    """Get BERV from FITS header ('HIERARCH CARACAL BERV') [km/s].

    Uses `fitsutils.read_header_keywords`.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    notfound :
        Value to return if BERV keyword not found in FITS header.
    ext : int, default 0
        Extension of the FITS from which to get the header. Only used if `filin` is a FITS file.
    outfmt : {'single', 'dict'}
        If 'dict', return a dictionary with the header keyword and its value. If 'single', return only the value.

    """
    kw = 'HIERARCH CARACAL BERV'
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def caracal_drift(filin, notfound=np.nan, ext=0, outfmt='single'):
    """Get drift and drift error from FITS header ('HIERARCH CARACAL DRIFT FP RV', 'HIERARCH CARACAL DRIFT FP E_RV') [m/s].

    Uses `fitsutils.read_header_keywords`.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    notfound :
        Value to return if keyword not found in FITS header.
    ext : int, default 0
        Extension of the FITS from which to get the header. Only used if `filin` is a FITS file.
    outfmt : {'single', 'dict'}
        If 'dict', return a dictionary with the header keyword and its value. If 'single', return only the values: drift, drifterr.
    """
    kws = ['HIERARCH CARACAL DRIFT FP RV', 'HIERARCH CARACAL DRIFT FP E_RV']
    dic = fitsutils.read_header_keywords(filin, kws, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kws[0]], dic[kws[1]]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def caracal_bjd(filin, notfound=np.nan, ext=0, outfmt='single', name='bjd'):
    """Get BJD from FITS header ('HIERARCH CARACAL BJD').
    Add  + 2400000 to value in the header.

    Uses `fitsutils.read_header_keywords`.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    notfound :
        Value to return if BERV keyword not found in FITS header.
    ext : int, default 0
        Extension of the FITS from which to get the header. Only used if `filin` is a FITS file.
    outfmt : {'single', 'dict'}
        If 'dict', return a dictionary with the header keyword and its value. If 'single', return only the value.
    name : str
    """
    kw = 'HIERARCH CARACAL BJD'
    if name is not None: names = {kw: name}
    else: names = None
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext, names=names)
    if name is not None: dic[name] = dic[name] + 2400000.
    else: dic[kw] = dic[kw] + 2400000.
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def caracal_bjd_lisobs(lisobs, notfound=np.nan, ext=0, name='bjd'):
    """Get BJD from FITS header ('HIERARCH CARACAL BJD') for the observations in `lisobs`. Add  + 2400000 to value in the header.

    Returns
    -------
    data : pandas dataframe
    """
    kw = 'HIERARCH CARACAL BJD'
    if name is not None: names = {kw: name}
    else: names = None
    data = fitsutils.read_header_keywords_lisobs(lisobs, kw, notfound=notfound, ext=ext, names=names)
    if name is not None: data[name] = data[name] + 2400000.
    else: data[kw] = data[kw] + 2400000.
    return data


def caracal_rvabs(filin, notfound=np.nan, ext=0, outfmt='single'):
    """Get the absolute RV and its error from FITS header ('HIERARCH CARACAL SERVAL RV', 'HIERARCH CARACAL SERVAL E_RV') [km/s].

    Uses `fitsutils.read_header_keywords`.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    notfound :
        Value to return if keyword not found in FITS header.
    ext : int, default 0
        Extension of the FITS from which to get the header. Only used if `filin` is a FITS file.
    outfmt : {'single', 'dict'}
        If 'dict', return a dictionary with the header keyword and its value. If 'single', return only the values: rvabs, rvabserr.
    """
    kws = ['HIERARCH CARACAL SERVAL RV', 'HIERARCH CARACAL SERVAL E_RV']
    dic = fitsutils.read_header_keywords(filin, kws, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kws[0]], dic[kws[1]]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def caracal_rvabs_lisobs(lisobs, notfound=np.nan, ext=0):
    """Get the absolute RV and its error from FITS header ('HIERARCH CARACAL SERVAL RV', 'HIERARCH CARACAL SERVAL E_RV') [km/s] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kws = ['HIERARCH CARACAL SERVAL RV', 'HIERARCH CARACAL SERVAL E_RV']
    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext)
    return data


def caracal_rvcorrection(filin, use_berv=True, use_drift=True, notfound=np.nan):  # , ext=0, outfmt=single):
    """Get RV correction for a given observation from CARACAL reduced FITS header.
    To apply the correction:
        wcorrected = w * (1 - shift/c)

    See also `serval_caracal_rvcorrection`.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    use_berv, use_drift : bool, default True
        Consider or not the BERV and the drift corrections

    Returns
    -------
    shift, shifterr : float
        RV corrections [m/s]
    dicshift : dict
    """

    # Read header
    if isinstance(filin, str): header = fitsutils.read_header(filin)
    elif isinstance(filin, fits.header.Header): header = filin

    if use_berv:
        berv = caracal_berv(header, notfound=notfound)*1.e3  # [m/s]
        berverr = np.nan
    else:
        berv = np.nan
        berverr = np.nan

    if use_drift:
        drift, drifterr = caracal_drift(header, notfound=notfound)  # [m/s]
    else:
        drift = np.nan
        drifterr = np.nan

    dicshift = {'berv': berv, 'drift': drift}
    dicshift_correctsign = {'berv': - berv, 'drift': + drift}
    dicshifterr = {'berverr': berverr, 'drifterr': drifterr}

    shift, shifterr = 0., 0.
    for k, v in dicshift_correctsign.items():
        if np.isfinite(v):
            shift += v
        else:
            print('Warning: {}={}. No {} correction applied!'.format(k, v, k))
    for k, v in dicshifterr.items():
        if np.isfinite(v):
            shifterr += v**2
        else:
            if k != 'berverr':
                print('Warning: {}={}. No {} correction error applied!'.format(k, v, k))
    shifterr = np.sqrt(shifterr)

    dicshift.update(dicshifterr)  # Merge dictionaries
    dicshift['shift'] = shift
    dicshift['shifterr'] = shifterr

    return shift, shifterr, dicshift


def caracal_rvcorrection_lisobs(lisobs, use_berv=True, use_drift=True, notfound=np.nan):
    """Get RV correction of all the observations in `lisobs` from their FITS header.

    If a value is a nan, it is considered a 0.

    To apply the correction:
        wcorrected = w * (1 - shift/c)
    where
        shift = - berv + drift

    See also `serval_rvcorrection_lisobs`.

    Parameters
    ----------
    lisobs : list of str
        List of the reduced FITS files to be used.
    use_berv, use_drift: bool, default True
        Consider or not the BERV and drift corrections.

    Returns
    -------
    shift : pandas dataframe
        Columns: timeid, shift
    shifterr : pandas dataframe
        Columns: timeid, shifterr
    dataused : pandas dataframe
        All data used to compute the shift and it error. Columns: timeid, RVs used, shift, shifterr
    """

    lisdicshift = []
    for obs in lisobs:
        shift, shifterr, dicshift = caracal_rvcorrection(obs, use_berv=True, use_drift=True, notfound=notfound)
        dicshift['obs'] = obs
        dicshift['timeid'] = os.path.basename(obs).replace('_A.fits', '.fits')
        lisdicshift.append(dicshift)
    datashift = pd.DataFrame(lisdicshift)
    datashift.set_index('timeid', inplace=True)

    shift = datashift['shift']
    shifterr = datashift['shifterr']
    return shift, shifterr, datashift


def caracal_snr(filin, ords=None, notfound=np.nan, ext=0):
    """Get the SNR of the orders in `ords` (e.g. from keyword 'HIERARCH CARACAL FOX SNR 36').

    Uses:
        - `fitsutils.read_header`
        - `fitsutils.read_header_keywords`

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
    if ords is None:  # SNR of all orders
        pattern = '*CARACAL FOX SNR*'
        # Read header
        if isinstance(filin, str): header = fitsutils.read_header(filin, ext=ext)
        elif isinstance(filin, fits.header.Header): header = filin
        # Get SNR for all orders
        kws = header[pattern]  # astropy.io.fits.header.Header object
        vals = list(kws.values())
        kws = list(kws.keys())
        dic = dict(zip(kws, vals))  # dict object

    else:  # SNR of specific orders in `ords`
        # Make sure ords is a list-type of ints
        if not isinstance(ords, (list, tuple, np.ndarray)): ords = [ords]
        if not isinstance(ords[0], int): ords = [int(o) for o in ords]
        kws = ['HIERARCH CARACAL FOX SNR {:d}'.format(o) for o in ords]
        dic = fitsutils.read_header_keywords(filin, kws, notfound=notfound, ext=ext)

    return dic


def caracal_snr_lisobs(lisobs, ords=None, notfound=np.nan, ext=0):
    """Get the SNR of the orders in `ords` (e.g. from keyword 'HIERARCH CARACAL FOX SNR 36') for the observations in `lisobs`.

    Uses:
        - `fitsutils.read_header`
        - `fitsutils.read_header_keywords_lisobs`

    Returns
    -------
    data : pandas dataframe
    """
    if ords is None:  # SNR of all orders
        pattern = '*CARACAL FOX SNR*'
        # Get keywords: Read header of 1st observation
        filin = lisobs[0]
        if isinstance(filin, str): header = fitsutils.read_header(filin, ext=ext)
        elif isinstance(filin, fits.header.Header): header = filin
        # Get SNR for all orders
        kws = header[pattern]  # astropy.io.fits.header.Header object
        kws = list(kws.keys())

    else:  # SNR of specific orders in `ords`
        # Make sure ords is a list-type of ints
        if not isinstance(ords, (list, tuple, np.ndarray)): ords = [ords]
        if not isinstance(ords[0], int): ords = [int(o) for o in ords]
        kws = ['HIERARCH CARACAL FOX SNR {:d}'.format(o) for o in ords]

    # Check that HIERARCH in keywords
    if not kws[0].startswith('HIERARCH'):
        kws = ['HIERARCH ' + i if not i.startswith('HIERARCH') else i for i in kws]

    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext)
    return data


def caracal_ron(filin, notfound=np.nan, ext=0, outfmt='single'):
    """Get readout noise from FITS header ('E_READN1') [electrons].

    Uses `fitsutils.read_header_keywords`.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    notfound :
        Value to return if BERV keyword not found in FITS header.
    ext : int, default 0
        Extension of the FITS from which to get the header. Only used if `filin` is a FITS file.
    outfmt : {'single', 'dict'}
        If 'dict', return a dictionary with the header keyword and its value. If 'single', return only the value.

    """
    kw = 'E_READN1'
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def caracal_ron_lisobs_vis(lisobs, notfound=np.nan, ext=0):
    """Get the readout noise from FITS header ('E_READN1') [electrons] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kws = ['E_READN1']
    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext)
    return data


def caracal_ron_lisobs_nir(lisobs, notfound=np.nan, ext=0):
    """Get the readout noise from FITS header ('E_READN') [electrons] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kws = ['E_READN']
    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext)
    return data


def caracal_exptime(filin, notfound=np.nan, ext=0, outfmt='single'):
    """Get exposure time from FITS header ('EXPTIME') [s].

    Uses `fitsutils.read_header_keywords`.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    notfound :
        Value to return if BERV keyword not found in FITS header.
    ext : int, default 0
        Extension of the FITS from which to get the header. Only used if `filin` is a FITS file.
    outfmt : {'single', 'dict'}
        If 'dict', return a dictionary with the header keyword and its value. If 'single', return only the value.

    """
    kw = 'EXPTIME'
    dic = fitsutils.read_header_keywords(filin, kw, notfound=notfound, ext=ext)
    if outfmt == 'single': return dic[kw]
    elif outfmt == 'dict': return dic
    else: sys.exit('`outfmt`={} not valid!'.format(outfmt))


def caracal_exptime_lisobs(lisobs, notfound=np.nan, ext=0):
    """Get the exposure time from FITS header ('EXPTIME') [s] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kws = ['EXPTIME']
    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext)
    return data


def caracal_airmass_lisobs(lisobs, notfound=np.nan, ext=0):
    """Get the readout noise from FITS header ('EXPTIME') [s] for the observations in `lisobs`.

    Returns
    -------
    data : pandas dataframe
    """
    kws = ['AIRMASS']
    data = fitsutils.read_header_keywords_lisobs(lisobs, kws, notfound=notfound, ext=ext)
    return data

###############################################################################

# -----------------------------------------------------------------------------
#
# SERVAL utils
#
# -----------------------------------------------------------------------------

# SERVAL template
# ---------------

def serval_tpl_read(filin, ln=False):
    """
    Read SERVAL template.

    Transform wavelength from natural log scale to linear scale if `ln` is `False` (default). If not, the returned wavelength is in natural log scale.

    Parameters
    ----------
    filin : str
        Full path to SERVAL template file.
    ls : bool, optional
        Return the wavelength in ln scale or not. Default: False.

    Returns
    -------
    w
    f
    sf
    header

    Notes
    -----
    SERVAL template FITS format:
        Primary header (mostly copied information from the highest SNR spectra)
        1st extension [SPEC] - the coadded spectrum
        2nd extension [SIG] - error estimate for SPEC
        3rd extension [WAVE] - (uniform) logarithm wavelength

    Wavelength is given in natural logarithm.
    Wavelength scale origins from the observation which was used as start guess (given in the FILENAME or DATE-OBS keyword; usually the highest S/N spectrum; its full header is copied to the template).
    Wavelength scale was corrected for the barycentric motion as given in the HIERARCH SERVAL BERV keyword.
    HIERARCH CARACAL SERVAL RV might serve as an estimate for the absolute RV of the template.
    """

    # Read FITS
    with fits.open(filin) as hdulist:
        header = hdulist[0].header
        wln = hdulist['wave'].data
        f = hdulist['spec'].data
        sf = hdulist['sig'].data

    # Wavelength scale
    if ln:
        w = wln
    else:  # SERVAL template wavelength in natural log -> Change to linear
        w = np.asarray([np.exp(wln_i) for wln_i in wln])
    return w, f, sf, header


def serval_tpl_nobs(filin, notfound=np.nan):
    """Get number of observations used to create the template (from header keyword 'HIERARCH SERVAL COADD NUM')."""
    data = fitsutils.read_header_keywords(filin, 'HIERARCH SERVAL COADD NUM', notfound=notfound)
    nobs = data['HIERARCH SERVAL COADD NUM']
    return nobs


def serval_tpl_berv(filin, notfound=np.nan):
    """Get BERV used to correct the template (from header keyword 'HIERARCH SERVAL BERV' or 'HIERARCH CARACAL BERV' if the other one is not present) [km/s]."""
    try: berv = fitsutils.read_header_keywords(filin, 'HIERARCH SERVAL BERV', notfound=notfound)['HIERARCH SERVAL BERV']
    except: berv = fitsutils.read_header_keywords(filin, 'HIERARCH CARACAL BERV', notfound=notfound)['HIERARCH CARACAL BERV']
    return berv


# SERVAL outputs
# --------------

def read_file2dataframe(fil, column_names, sep='\s+', header=None, index_col=0, ifnofilout='empty', nrow=1):
    """Read data from `fil` into pandas DataFrame.

    Parameters
    ----------
    ifnofilout : {'empty', 'nan', 'dfnan'}
        If input file does not exist, there are 3 return options:
        1) 'empty': return an empty DataFrame with the corresponding columns indicated in `column_names`.
        2) 'nan': return `np.nan`.
        3) 'dfnan': return a DataFrame filled with `np.nan` with the corresponding columns indicated in `column_names`. The number of rows is indicated with `nrow`. If `nrow` is smaller than 1, it will be changed to 1. Use `ifnofilout='empty'` if want dataframe with no rows.
        If none of the above, exit.

    Returns
    -------
    data : pd.Dataframe or specified with `ifnofilout`.
    """
    try:
        data = pd.read_csv(fil, sep=sep, header=header, names=column_names, index_col=index_col)
    # If file does not exist
    except:
        if ifnofilout == 'empty':
            data = pd.DataFrame(columns=column_names)
            data.set_index(column_names[index_col], inplace=True)
        elif ifnofilout == 'nan':
            data = np.nan
        elif ifnofilout == 'dfnan':
            if nrow < 1: nrow = 1
            r = [np.nan] * int(nrow)
            data = {}
            for c in column_names: data[c] = r
            data = pd.DataFrame(data, index=data[column_names[index_col]])
        else:
            sys.exit('Cannot read {}. Exit.'.format(fil))
    return data


def serval_read_rvc(fil, SI=True, ifnofilout='empty', nrow=1):
    """
    SI : bool, default True
        Return data in the international system of units (m/s instead of km/s, etc)

    obj.rvc.dat    Radial velocity (drift and secular acceleration corrected, nan-drifts treated as zero)
    Description of file: obj.rvc.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       RVC       Radial velocity (drift and sa corrected)
         3 D      m/s     E_RVC       Radial velocity error
         4 D      m/s       DRIFT     CARACAL drift measure
         5 D      m/s     E_DRIFT     CARACAL drift measure error
         6 D      m/s       RV        Radial velocity
         7 D      m/s     E_RV        Radial velocity error
         8 D      km/s      BERV      Barycentric earth radial velocity [1]
         9 D      m/s       SADRIFT   Drift due to secular acceleration
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'servalrvc', 'servalrvcerr', 'servaldrift', 'servaldrifterr', 'servalrv', 'servalrverr', 'servalberv', 'servalsa']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    if SI:
        data['servalberv'] = data['servalberv'] * 1.e3  # [m/s]
    return data


def serval_read_info(fil, index='timeid', SI=True, ifnofilout='empty', nrow=1, oldversion=False):
    """
    obj.info.csv   Infomation from fits file
    Description of file: obj.info.csv
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 A      ---       TIMEID    Identifier for file (time stamp)
         2 D      ---       BJD       Barycentric Julian date [1]
         3 D      ---       SNREF     Signal-to-ratio in reference order (CARM_VIS: 36, CARM_NIR: 16, HARPS: 55)
         4 A      ---       OBJ       Fits keyword OBJECT
         5 D      s         EXPTIME   Exposure time
         6 A      ---       SPT       Spectral type SpT ccf.mask
         7 I256   ---       FLAG      Flag bad spectra/instrument mode (0: ok)
         8 D      ---       AIRMASS   Air mass from fits header
         9 D      deg       RA        Telescope Ascension (degrees)
        10 D      deg       DE        Telescope declination (degrees)
    --------------------------------------------------------------------------------
    
    Notes
    -----
    Columns description above not updated
    """
    if oldversion:
        column_names = ['timeid', 'bjd', 'bervunknown', 'snref', 'fitsobj', 'exptime', 'spt', 'flag', 'airmass', 'ratel', 'dectel']
    else:
        column_names = ['timeid', 'bjd', 'bervunknown', 'snref', 'fitsobj', 'exptime', 'spt', 'flag', 'airmass', 'ratel', 'dectel', 'unknown1', 'unknown2', 'unknown3']
    index_col = 0
    if index == 'timeid': index_col = 0
    elif index == 'bjd': index_col = 1
    data = read_file2dataframe(fil, column_names, sep=';', index_col=index_col, ifnofilout=ifnofilout, nrow=nrow)
    if SI:
        data['bervunknown'] = data['bervunknown'] * 1.e3  # [m/s]
    return data


def serval_read_brv(fil, ifnofilout='empty', nrow=1):
    """
    obj.brv.dat    Barycentric earth radial velocity
    Description of file: obj.brv.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      km/s      BERV      Barycentric earth radial velocity [1]
         3 D      ---       DRSBJD    Barycentric Julian date (CARACAL)
         4 D      km/s      DRSBERV   Barycentric earth radial velocity (CARACAL)
         5 D      m/s       DRSDRIFT  CARACAL drift measure in fib B
         6 A      ---       TIMEID    Identifier for file (time stamp)
         7 D      ---       TMMEAN    Flux weighted mean point
         8 D      s         EXPTIME   Exposure time
         9 D      km/s      BERVSTART BERV at start of exposure
        10 D      km/s      BERVEND   BERV at end of exposure
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'berv', 'drsbjd', 'drsberv', 'drsdrift', 'timeid', 'tmmean', 'exptime', 'bervstart', 'bervend']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def serval_read_srv(fil, ifnofilout='empty', nrow=1):
    """
    obj.srv.dat    Serval products time series compilation
    Description of file: obj.srv.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       RVC       Radial velocity (drift and sa corrected)
         3 D      m/s     E_RVC       Radial velocity error
         4 D      m/s       CRX       Chromatic index (Slope over logarithmic wavelength)
         5 D      m/s     E_CRX       error for CRX (slope error)
         6 D      m^2/s^2   DLW       Differential Line Width
         7 D      m^2/s^2 E_DLW       Error in DLW
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'servalrvc', 'servalrvcerr', 'servalcrx', 'servalcrxerr', 'servaldlw', 'servaldlwerr']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def serval_read_halpha(fil, ifnofilout='empty', nrow=1):
    """
    obj.halpha.dat Halpha line index (requires absolute RVs)
    Description of file: obj.halpha.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      ---       HALPHA    Halpha index (6562.808) (-40,40) km/s
         3 D      ---     E_HALPHA    HALPHA error
         4 D      ---       HACEN     Halpha mean flux
         5 D      ---     E_HACEN     HACEN error
         6 D      ---       HALEFT    Halpha reference region (-300,-100) km/s
         7 D      ---     E_HALEFT    HALEFT error
         8 D      ---       HARIGH    Halpha reference region (100,300) km/s
         9 D      ---     E_HARIGH    HARIGH error
        10 D      ---       CAI       Calcium I index
        11 D      ---     E_CAI       CAI error
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'servalhalpha', 'servalhalphaerr', 'servalhacen', 'servalhacenerr', 'servalhaleft', 'servalhalefterr', 'servalharight', 'servalharighterr', 'servalcai', 'servalcaierr']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def serval_read_cairt(fil, ifnofilout='empty', nrow=1):
    """
    obj.cairt.dat  CAII IRT line index (requires absolute RVs)
    Description of file: obj.cairt.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      ---       CAIRT1    CaII-IRT1 index (8498.02) (-15.,15.) km/s [2]  # NIST + my definition
         3 D      ---     E_CAIRT1    CAIRT1 error
         4 D      ---       CAIRT2    CaII-IRT2 index (8542.09) (-15.,15.) km/s [3]   # NIST + my definition
         5 D      ---     E_CAIRT2    CAIRT2 error
         6 D      ---       CAIRT3    CaII-IRT3 index (8662.14) (-15.,15.) km/s [4]   # NIST + my definition
         7 D      ---     E_CAIRT3    CAIRT3 error
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'servalcairt1', 'servalcairt1err', 'servalcairt2', 'servalcairt2err', 'servalcairt3', 'servalcairt3err']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def serval_read_nad(fil, ifnofilout='empty', nrow=1):
    """
    obj.nad.dat    NaD line index (requires absolute RVs)
    Description of file: obj.nad.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      ---       NAD1      NaD1 index (5889.950943) (-15,15) km/s [5]
         3 D      ---     E_NAD1      NAD1 error
         4 D      ---       NAD2      NaD2 index (5895.924237) (-15,15) km/s [6]
         5 D      ---     E_NAD2      NAD2 error
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'servalnad1', 'servalnad1err', 'servalnad2', 'servalnad2err']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def serval_read_rvo(fil, inst='CARM_VIS', ifnofilout='empty', nrow=1):
    """Get data from SERVAL `obj.rvo.dat`.

    Number of columns in file depends on the number of orders of the instrument.

    Parameters
    ----------
    fil : str
        Path to input file `obj.rvo.daterr`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN'}
        Specifiy indtrument to get the number of orders (necessary to know the number of columns). If none of the options above, try to get the number of orders from the file name or directly from the number of columns of the file `fil`.
    ifnofilout, nrow :
        Output options of function `read_file2dataframe`. See its documentation for more details.

    Returns
    -------
    data : pandas dataframe or np.nan
        See the documentation of the function `read_file2dataframe` for more details.

    obj.rvo.dat    Radial velocity (orderwise, not drift corrected)
    Description of file: obj.rvo.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       RV        Radial velocity (mean, not drift and not sa corrected)
         3 D      m/s     E_RV        Radial velocity error
         4 D      m/s       RVMED     Radial velocity (median)
         5 D      m/s     E_RVMED    Radial velocity (median) error
         6 D      m/s       RVO_00    Radial velocity in order 0
         7 D      m/s       RVO_01    Radial velocity in order 1
         8 D      m/s       RVO_02    Radial velocity in order 2
    etc...
    --------------------------------------------------------------------------------
    """

    # If instrument known, get the number of orders
    ords_known = False
    if inst == 'CARM_VIS':
        nord = 61
        ords_known = True
    elif inst == 'CARM_NIR':
        nord = 28
        ords_known = True
    elif inst == 'HARPS' or inst == 'HARPN':
        nord = 72
        ords_known = True

    # If no instrument specified, try to get the number of orders from the filename
    else:
        if 'vis' in fil or 'VIS' in fil:
            nord = 61
            ords_known = True
        elif 'nir' in fil or 'NIR' in fil:
            nord = 28  # don't know actually
            ords_known = True

    # -----------------------------------------------------

    # If number of orders known, read into pandas dataframe directly
    if ords_known:
        ords = np.arange(0, nord, 1)
        column_names = ['bjd', 'servalrv', 'servalrverr', 'servalrvmed', 'servalrvmederr'] + ['servalrvo{:02d}'.format(o) for o in ords]
        data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    # If number of orders not known, read the file and get it from there
    else:
        try:
            # Read data 1st to get number of orders
            data_raw = np.loadtxt(fil, unpack=True)
            nord = len(data_raw[5:])
            ords = np.arange(0, nord, 1)

            # Put data in pandas dataframe
            data_dic = {'bjd': data_raw[0], 'servalrv': data_raw[1], 'servalrverr': data_raw[2], 'servalrvmed': data_raw[3], 'servalrvmederr': data_raw[4]}
            dic_ords = {'servalrvo{:02d}'.format(i): data_raw[i+5] for i in ords}
            data_dic.update(dic_ords)  # Merge dictionaries
            data = pd.DataFrame(data_dic)
            data.set_index('bjd', inplace=True)
        # If cannot read the file, return the output of the function `read_file2dataframe` specified by `ifnofilout`
        except:
            ords = np.arange(0, 1, 1)
            column_names = ['bjd', 'servalrv', 'servalrverr', 'servalrvmed', 'servalrvmederr'] + ['servalrvo{:02d}'.format(o) for o in ords]
            data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    return data


def serval_read_rvoerr(fil, inst='CARM_VIS', ifnofilout='empty', nrow=1):
    """Get data from SERVAL `obj.e_rvo.dat`.

    Number of columns in file depends on the number of orders of the instrument.

    Parameters
    ----------
    fil : str
        Path to input file `obj.rvo.daterr`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN'}
        Specifiy indtrument to get the number of orders (necessary to know the number of columns). If none of the options above, try to get the number of orders from the file name or directly from the number of columns of the file `fil`.
    ifnofilout, nrow :
        Output options of function `read_file2dataframe`. See its documentation for more details.

    Returns
    -------
    data : pandas dataframe or np.nan
        See the documentation of the function `read_file2dataframe` for more details.

    Description of file: obj.rvo.daterr
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       RV        Radial velocity (mean, not drift and not sa corrected)
         3 D      m/s     E_RV        Radial velocity error
         4 D      m/s       RVMED     Radial velocity (median)
         5 D      m/s     E_RVMED     Radial velocity (median) error
         6 D      m/s     E_RVO_00    Radial velocity error in order 0
         7 D      m/s     E_RVO_01    Radial velocity error in order 1
         8 D      m/s     E_RVO_02    Radial velocity error in order 2
    etc...
    --------------------------------------------------------------------------------
    """

    # If instrument known, get the number of orders
    ords_known = False
    if inst == 'CARM_VIS':
        nord = 61
        ords_known = True
    elif inst == 'CARM_NIR':
        nord = 28
        ords_known = True
    elif inst == 'HARPS' or inst == 'HARPN':
        nord = 72
        ords_known = True

    # If no instrument specified, try to get the number of orders from the filename
    else:
        if 'vis' in fil or 'VIS' in fil:
            nord = 61
            ords_known = True
        elif 'nir' in fil or 'NIR' in fil:
            nord = 28
            ords_known = True

    # -----------------------------------------------------

    # If number of orders known, read into pandas dataframe directly
    if ords_known:
        ords = np.arange(0, nord, 1)
        column_names = ['bjd', 'servalrv', 'servalrverr', 'servalrvmed', 'servalrvmederr'] + ['servalrvo{:02d}err'.format(o) for o in ords]
        data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    # If number of orders not known, read the file and get it from there
    else:
        try:
            # Read data 1st to get number of orders
            data_raw = np.loadtxt(fil, unpack=True)
            nord = len(data_raw[5:])
            ords = np.arange(0, nord, 1)

            # Put data in pandas dataframe
            data_dic = {'bjd': data_raw[0], 'servalrv': data_raw[1], 'servalrverr': data_raw[2], 'servalrvmed': data_raw[3], 'servalrvmederr': data_raw[4]}    
            dic_ords = {'servalrvo{:02d}err'.format(i): data_raw[i+5] for i in ords}
            data_dic.update(dic_ords) # Merge dictionaries
            data = pd.DataFrame(data_dic)
            data.set_index('bjd', inplace=True)
        # If cannot read the file, return the output of the function `read_file2dataframe` specified by `ifnofilout`
        except:
            ords = np.arange(0, 1, 1)
            column_names = ['bjd', 'servalrv', 'servalrverr', 'servalrvmed', 'servalrvmederr'] + ['servalrvo{:02d}err'.format(o) for o in ords]
            data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    return data


def serval_read_crx(fil, inst='CARM_VIS', ifnofilout='empty', nrow=1):
    """Get data from SERVAL `obj.crx.dat`.

    Number of columns in file depends on the number of orders of the instrument.

    Parameters
    ----------
    fil : str
        Path to input file `obj.crx.dat`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN'}
        Specifiy indtrument to get the number of orders (necessary to know the number of columns). If none of the options above, try to get the number of orders from the file name or directly from the number of columns of the file `fil`.
    ifnofilout, nrow :
        Output options of function `read_file2dataframe`. See its documentation for more details.

    Returns
    -------
    data : pandas dataframe or np.nan
        See the documentation of the function `read_file2dataframe` for more details.

    obj.crx.dat    Chromatic RV index (wavelength depedence of the RV)
    Description of file: obj.crx.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       CRX       Chromatic index (Slope over logarithmic wavelength)
         3 D      m/s     E_CRX       Slope error
         4 D      m/s       CRX_OFF   Offset parameter alpha
         5 D      m/s     E_CRX_OFF   Error for offset parameter alpha
         6 D      A         L_V       Wavelength at which the slope intersects RV
         7 D                LNL_00    Central logarithmic wavelength of aperture 1
         8 D                LNL_01    Central logarithmic wavelength of aperture 2
    etc...
    --------------------------------------------------------------------------------
    """

    # If instrument known, get the number of orders
    ords_known = False
    if inst == 'CARM_VIS':
        nord = 61
        ords_known = True
    elif inst == 'CARM_NIR':
        nord = 28
        ords_known = True
    elif inst == 'HARPS' or inst == 'HARPN':
        nord = 72
        ords_known = True

    # If no instrument specified, try to get the number of orders from the filename
    else:
        if 'vis' in fil or 'VIS' in fil:
            nord = 61
            ords_known = True
        elif 'nir' in fil or 'NIR' in fil:
            nord = 28  # don't know actually
            ords_known = True

    # -----------------------------------------------------

    # If number of orders known, read into pandas dataframe directly
    if ords_known:
        ords = np.arange(0, nord, 1)
        column_names = ['bjd', 'servalcrx', 'servalcrxerr', 'servalcrxoff', 'servalcrxofferr', 'lv'] + ['servallnlo{:02d}'.format(o) for o in ords]
        data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    # If number of orders not known, read the file and get it from there
    else:
        try:
            # Read data 1st to get number of orders
            data_raw = np.loadtxt(fil, unpack=True)
            nord = len(data_raw[6:])
            ords = np.arange(0, nord, 1)

            # Put data in pandas dataframe
            data_dic = {'bjd': data_raw[0], 'servalcrx': data_raw[1], 'servalcrxerr': data_raw[2], 'servalcrxoff': data_raw[3], 'servalcrxofferr': data_raw[4], 'lv': data_raw[5]}
            dic_ords = {'servallnlo{:02d}'.format(i): data_raw[i+6] for i in ords}
            data_dic.update(dic_ords)  # Merge dictionaries
            data = pd.DataFrame(data_dic)
            data.set_index('bjd', inplace=True)
        # If cannot read the file, return the output of the function `read_file2dataframe` specified by `ifnofilout`
        except:
            ords = np.arange(0, 1, 1)
            column_names = ['bjd', 'servalcrx', 'servalcrxerr', 'servalcrxoff', 'servalcrxofferr', 'lv'] + ['servallnlo{:02d}'.format(o) for o in ords]
            data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    return data


def serval_read_dlw(fil, inst='CARM_VIS', ifnofilout='empty', nrow=1):
    """Get data from SERVAL `obj.dlw.dat`.

    Number of columns in file depends on the number of orders of the instrument.

    Parameters
    ----------
    fil : str
        Path to input file `obj.dlw.dat`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN'}
        Specifiy indtrument to get the number of orders (necessary to know the number of columns). If none of the options above, try to get the number of orders from the file name or directly from the number of columns of the file `fil`.
    ifnofilout, nrow :
        Output options of function `read_file2dataframe`. See its documentation for more details.

    Returns
    -------
    data : pandas dataframe or np.nan
        See the documentation of the function `read_file2dataframe` for more details.

    obj.dlw.dat    Differential FWHM
    Description of file: obj.dlw.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      ---       dLW       differential line width
         3 D      ---     E_dLW       error for dLW
         4 D      ---       dLWO_00   differential line width in order 0
         5 D      ---       dLWO_01   differential line width in order 1
         6 D      ---       dLWO_02   differential line width in order 2
    etc...
    --------------------------------------------------------------------------------
    """

    # If instrument known, get the number of orders
    ords_known = False
    if inst == 'CARM_VIS':
        nord = 61
        ords_known = True
    elif inst == 'CARM_NIR':
        nord = 28
        ords_known = True
    elif inst == 'HARPS' or inst == 'HARPN':
        nord = 72
        ords_known = True

    # If no instrument specified, try to get the number of orders from the filename
    else:
        if 'vis' in fil or 'VIS' in fil:
            nord = 61
            ords_known = True
        elif 'nir' in fil or 'NIR' in fil:
            nord = 28  # don't know actually
            ords_known = True

    # -----------------------------------------------------

    # If number of orders known, read into pandas dataframe directly
    if ords_known:
        ords = np.arange(0, nord, 1)
        column_names = ['bjd', 'servaldlw', 'servaldlwerr'] + ['servaldlwo{:02d}'.format(o) for o in ords]
        data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    # If number of orders not known, read the file and get it from there
    else:
        try:
            # Read data 1st to get number of orders
            data_raw = np.loadtxt(fil, unpack=True)
            nord = len(data_raw[3:])
            ords = np.arange(0, nord, 1)

            # Put data in pandas dataframe
            data_dic = {'bjd': data_raw[0], 'servaldlw': data_raw[1], 'servaldlwerr': data_raw[2]}
            dic_ords = {'servaldlwo{:02d}'.format(i): data_raw[i+3] for i in ords}
            data_dic.update(dic_ords)  # Merge dictionaries
            data = pd.DataFrame(data_dic)
            data.set_index('bjd', inplace=True)
        # If cannot read the file, return the output of the function `read_file2dataframe` specified by `ifnofilout`
        except:
            ords = np.arange(0, 1, 1)
            column_names = ['bjd', 'servaldlw', 'servaldlwerr'] + ['servaldlwo{:02d}'.format(o) for o in ords]
            data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    return data


def serval_read_dlwerr(fil, inst='CARM_VIS', ifnofilout='empty', nrow=1):
    """Get data from SERVAL `obj.e_dlw.dat`.

    Number of columns in file depends on the number of orders of the instrument.

    Parameters
    ----------
    fil : str
        Path to input file `obj.e_dlw.daterr`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN'}
        Specifiy indtrument to get the number of orders (necessary to know the number of columns). If none of the options above, try to get the number of orders from the file name or directly from the number of columns of the file `fil`.
    ifnofilout, nrow :
        Output options of function `read_file2dataframe`. See its documentation for more details.

    Returns
    -------
    data : pandas dataframe or np.nan
        See the documentation of the function `read_file2dataframe` for more details.
    """

    # If instrument known, get the number of orders
    ords_known = False
    if inst == 'CARM_VIS':
        nord = 61
        ords_known = True
    elif inst == 'CARM_NIR':
        nord = 28
        ords_known = True
    elif inst == 'HARPS' or inst == 'HARPN':
        nord = 72
        ords_known = True

    # If no instrument specified, try to get the number of orders from the filename
    else:
        if 'vis' in fil or 'VIS' in fil:
            nord = 61
            ords_known = True
        elif 'nir' in fil or 'NIR' in fil:
            nord = 28  # don't know actually
            ords_known = True

    # -----------------------------------------------------

    # If number of orders known, read into pandas dataframe directly
    if ords_known:
        ords = np.arange(0, nord, 1)
        column_names = ['bjd', 'servaldlw', 'servaldlwerr'] + ['servaldlwo{:02d}err'.format(o) for o in ords]
        data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    # If number of orders not known, read the file and get it from there
    else:
        try:
            # Read data 1st to get number of orders
            data_raw = np.loadtxt(fil, unpack=True)
            nord = len(data_raw[3:])
            ords = np.arange(0, nord, 1)

            # Put data in pandas dataframe
            data_dic = {'bjd': data_raw[0], 'servaldlw': data_raw[1], 'servaldlwerr': data_raw[2]}
            dic_ords = {'servaldlwo{:02d}err'.format(i): data_raw[i+3] for i in ords}
            data_dic.update(dic_ords)  # Merge dictionaries
            data = pd.DataFrame(data_dic)
            data.set_index('bjd', inplace=True)
        # If cannot read the file, return the output of the function `read_file2dataframe` specified by `ifnofilout`
        except:
            ords = np.arange(0, 1, 1)
            column_names = ['bjd', 'servaldlw', 'servaldlwerr'] + ['servaldlwo{:02d}err'.format(o) for o in ords]
            data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    return data


def serval_read_snr(fil, inst='CARM_VIS', ifnofilout='empty', nrow=1):
    """Get data from SERVAL `obj.snr.dat`.

    Number of columns in file depends on the number of orders of the instrument.

    Parameters
    ----------
    fil : str
        Path to input file `obj.rvo.daterr`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN'}
        Specifiy indtrument to get the number of orders (necessary to know the number of columns). If none of the options above, try to get the number of orders from the file name or directly from the number of columns of the file `fil`.
    ifnofilout, nrow :
        Output options of function `read_file2dataframe`. See its documentation for more details.

    Returns
    -------
    data : pandas dataframe or np.nan
        See the documentation of the function `read_file2dataframe` for more details.

    obj.snr.dat    Signal-to-noise (orderwise)
    Description of file: obj.snr.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      ---       SNR       Overall signal to noise
         3 D      ---       SNRO_00   Signal to noise in order 0
         4 D      ---       SNRO_01   Signal to noise in order 1
         5 D      ---       SNRO_02   Signal to noise in order 2
    etc...
    --------------------------------------------------------------------------------
    """

    # If instrument known, get the number of orders
    ords_known = False
    if inst == 'CARM_VIS':
        nord = 61
        ords_known = True
    elif inst == 'CARM_NIR':
        nord = 28
        ords_known = True
    elif inst == 'HARPS' or inst == 'HARPN':
        nord = 72
        ords_known = True

    # If no instrument specified, try to get the number of orders from the filename
    else:
        if 'vis' in fil or 'VIS' in fil:
            nord = 61
            ords_known = True
        elif 'nir' in fil or 'NIR' in fil:
            nord = 28  # don't know actually
            ords_known = True

    # -----------------------------------------------------

    # If number of orders known, read into pandas dataframe directly
    if ords_known:
        ords = np.arange(0, nord, 1)
        column_names = ['bjd', 'servalsnr'] + ['servalsnro{:02d}'.format(o) for o in ords]
        data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    # If number of orders not known, read the file and get it from there
    else:
        try:
            # Read data 1st to get number of orders
            data_raw = np.loadtxt(fil, unpack=True)
            nord = len(data_raw[2:])
            ords = np.arange(0, nord, 1)

            # Put data in pandas dataframe
            data_dic = {'bjd': data_raw[0], 'servalsnr': data_raw[1]}
            dic_ords = {'servalsnro{:02d}'.format(i): data_raw[i+2] for i in ords}
            data_dic.update(dic_ords)  # Merge dictionaries
            data = pd.DataFrame(data_dic)
            data.set_index('bjd', inplace=True)
        # If cannot read the file, return the output of the function `read_file2dataframe` specified by `ifnofilout`
        except:
            ords = np.arange(0, 1, 1)
            column_names = ['bjd', 'servalsnr'] + ['servalsnro{:02d}'.format(o) for o in ords]
            data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    return data


def serval_read_chi(fil, inst='CARM_VIS', ifnofilout='empty', nrow=1):
    """Get data from SERVAL `obj.chi.dat`.

    Number of columns in file depends on the number of orders of the instrument.

    Parameters
    ----------
    fil : str
        Path to input file `obj.rvo.daterr`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN'}
        Specifiy indtrument to get the number of orders (necessary to know the number of columns). If none of the options above, try to get the number of orders from the file name or directly from the number of columns of the file `fil`.
    ifnofilout, nrow :
        Output options of function `read_file2dataframe`. See its documentation for more details.

    Returns
    -------
    data : pandas dataframe or np.nan
        See the documentation of the function `read_file2dataframe` for more details.

    obj.chi.dat    sqrt(reduced chi^2) (orderwise)
    Description of file: obj.chi.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      ---       RCHI      Overall reduced chi^2
         3 D      ---       RCHIO_00  Sqrt(reduced chi^2) in order 0
         4 D      ---       RCHIO_01  Sqrt(reduced chi^2) in order 1
         5 D      ---       RCHIO_02  Sqrt(reduced chi^2) in order 2
    etc...
    --------------------------------------------------------------------------------
    """

    # If instrument known, get the number of orders
    ords_known = False
    if inst == 'CARM_VIS':
        nord = 61
        ords_known = True
    elif inst == 'CARM_NIR':
        nord = 28
        ords_known = True
    elif inst == 'HARPS' or inst == 'HARPN':
        nord = 72
        ords_known = True

    # If no instrument specified, try to get the number of orders from the filename
    else:
        if 'vis' in fil or 'VIS' in fil:
            nord = 61
            ords_known = True
        elif 'nir' in fil or 'NIR' in fil:
            nord = 28  # don't know actually
            ords_known = True

    # -----------------------------------------------------

    # If number of orders known, read into pandas dataframe directly
    if ords_known:
        ords = np.arange(0, nord, 1)
        column_names = ['bjd', 'servalchi'] + ['servalchio{:02d}'.format(o) for o in ords]
        data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    # If number of orders not known, read the file and get it from there
    else:
        try:
            # Read data 1st to get number of orders
            data_raw = np.loadtxt(fil, unpack=True)
            nord = len(data_raw[2:])
            ords = np.arange(0, nord, 1)

            # Put data in pandas dataframe
            data_dic = {'bjd': data_raw[0], 'servalchi': data_raw[1]}
            dic_ords = {'servalchio{:02d}'.format(i): data_raw[i+2] for i in ords}
            data_dic.update(dic_ords)  # Merge dictionaries
            data = pd.DataFrame(data_dic)
            data.set_index('bjd', inplace=True)
        # If cannot read the file, return the output of the function `read_file2dataframe` specified by `ifnofilout`
        except:
            ords = np.arange(0, 1, 1)
            column_names = ['bjd', 'servalchi'] + ['servalchio{:02d}'.format(o) for o in ords]
            data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)

    return data


def serval_read_mlc(fil, ifnofilout='empty', nrow=1):
    """
    obj.mlc.dat    RV and CRX averages via maximum likelihood maps (experimental)
    Description of file: obj.mlc.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       MLRVC     ML Radial velocity (drift and sa corrected)
         3 D      m/s     E_MLRVC     ML Radial velocity error
         4 D      m/s       MLCRX     ML Chromatic index (Slope over logarithmic wavelength)
         5 D      m/s     E_MLCRX     error for MLCRX (slope error)
         6 D      m^2/s^2   DLW       Differential Line Width
         7 D      m^2/s^2 E_DLW       Error in DLW
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'servalmlrvc', 'servalmlrvcerr', 'servalmlcrx', 'servalmlcrxerr', 'servalmldlw', 'servalmldlwerr']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def serval_lisobs(listimeid):
    """Get observation filename, i.e. add change ".fits" to "_A.fits" in `timeid` column.
    """
    lisobs = [i.replace('.fits', '_A.fits') for i in listimeid]
    data = pd.DataFrame({'obs': lisobs}, index=listimeid)
    return data


def serval_get(dirin, obj=None, lisdataid=['rvc', 'info', 'brv', 'srv', 'halpha', 'cairt', 'nad', 'rvo', 'rvoerr', 'crx', 'dlw', 'dlwerr', 'snr', 'chi', 'mlc'], obs=True, inst=None, join='outer', ifnofilout='empty', nrow=1):
    """Read data from SERVAL outputs specified in the list `lisdataid` and return a pandas dataframe with all the data marged.

    Output data in SI (e.g. m/s instead of km/s).

    Parameters
    ----------
    dirin : str
        Directory containing the SERVAL outputs to be read.
    obj : str
        Tag of the SERVAL outputs, e.g. `obj.rvc.dat`. If None (default), try to get it from the files in `dirin` directly.
    lisdataid : list
        List with the id of the files to be read. E.g. 'rvc' for 'obj.rvc.dat', 'rvo' for 'obj.rvo.dat', 'rvoerr' for 'obj.rvo.daterr'
    obs : bool (default True)
        Get observation names (i.e. timeid with the "_A"). Useful to compare with other datasets. Only works if 'info' is in `lisdataid`.
    inst : {'CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN', None (default)}
        Instrument id for the data coming from files with order data (number of orders is instrument-depending). If None (default), try to get it from the file names.
    join : {'inner', 'outer'}
        How to join the pandas dataframes coming from different files if different rows present.
    ifnofilout, nrow : Output format.
    """

    # If obj not specified, get it from file names
    if obj is None:
        if os.path.exists(glob.glob(os.path.join(dirin, '*.info.csv'))[0]):
            obj = glob.glob(os.path.join(dirin, '*.info.csv'))[0].replace('.info.csv', '')
        else:
            sys.exit('File {} does not exist'.format(os.path.join(dirin, '*.info.csv')))

    # Read files
    data_lis = []
    for dataid in lisdataid:

        # File to be read
        if 'info' in dataid: filin = os.path.join(dirin, '{}.info.csv'.format(obj))
        # elif 'err' in dataid: filin = os.path.join(dirin, '{}.{}.daterr'.format(obj, dataid.replace('err', '')))
        elif 'err' in dataid: filin = os.path.join(dirin, '{}.e_{}.dat'.format(obj, dataid.replace('err', '')))
        else: filin = os.path.join(dirin, '{}.{}.dat'.format(obj, dataid))

        # Function to be used to read the file
        method_name = 'serval_read_{}'.format(dataid)
        method = globals().get(method_name)

        # Read data
        if method is not None:
            # Different parameters for different methods
            if dataid == 'info':
                info_index = 'bjd'
                data_lis.append(method(filin, index=info_index, ifnofilout=ifnofilout, nrow=nrow))
            elif dataid in ['serval_read_rvo', 'serval_read_rvoerr', 'serval_read_crx', 'serval_read_dlw', 'serval_read_dlwerr', 'serval_read_snr', 'serval_read_chi']:
                data_lis.append(method(filin, inst=inst, ifnofilout=ifnofilout, nrow=nrow))
            else:
                data_lis.append(method(filin, ifnofilout=ifnofilout, nrow=nrow))

            # Round BJD to be able to compare the different files
            if len(data_lis[-1]) > 1:
                ##### IMPORTANT CHANGE HERE
                data_lis[-1].index = np.round(data_lis[-1].index-2400000., decimals=5) + 2400000
                data_lis[-1]['bjd'] = data_lis[-1].index
                data_lis[-1]['bjdstr'] = ['{:.4f}'.format(i) for i in data_lis[-1]['bjd']]
                data_lis[-1].set_index('bjdstr', inplace=True)
        # if not method: # same as `if method is None`
        else:
            raise NotImplementedError('Method {} not implemented.'.format(method_name))

    # Merge dataframes into a single one
    data = pd.concat(data_lis, axis=1, join=join)
    data['bjdstr'] = data.index
    # Remove duplicates
    data = data.loc[:, ~data.columns.duplicated()]
    data.set_index('bjd', inplace=True)

    # Remove duplicated columns (rvc...)
    data = data.loc[:, ~data.columns.duplicated()]

    # Get observation names
    if obs and 'info' in dataid:

        # Make sure timeid is string
        # if obj == "J11477+008":
        mask = [type(d) == str for d in data['timeid']]
        data = data[mask]

        data['obs'] = serval_lisobs(data['timeid'].values).values

    return data


# SERVAL outputs all objects

def serval_lisobj(dirin, pattern='J*/'):
    """Same as `caracal_lisobj`"""
    return caracal_lisobj(dirin, pattern=pattern)


def serval_get_allobj(dirin, lisobj=None, patternobj='J*/', lisdataid=['rvc', 'info'], obs=True, inst=None, join='outer', ifnofilout='empty', nrow=1):
    """Get all serval data for a list of objects, i.e. run `serval_get` for a list of objects.

    See `serval_get` for more info.

    """
    if lisobj is None:
        lisdirobj, lisobj = serval_lisobj(dirin, pattern=patternobj)

    dataserval = {}
    for obj in lisobj:
        dataserval[obj] = serval_get(os.path.join(dirin, obj), obj=obj, lisdataid=lisdataid, inst=inst, join=join, ifnofilout=ifnofilout, nrow=nrow)

    return dataserval


# Getting data from SERVAL outputs

def serval_caracal_rvcorrection(filobs, servalin, obj=None, use_caracal=True, caracalin=None, use_berv=True, use_drift=True, use_sa=True, notfound=np.nan, ext=0, verb=False):
    """Get RV correction for a given observation from SERVAL data and optionally, if not available, from CARACAL reduced FITS header.

    To apply the correction:
        wcorrected = w * (1 - shift/c)

    Parameters
    ----------
    filobs : str
        Reduced CARACAL file name of the observations for which to get the RV correction.
    servalin : str or pd.DataFrame
        Path to directory containing SERVAL data or pandas dataframe with the necessary data (berv, drift and sa) already loaded.
    use_caracal : bool, default True
        Use value from CARACAL header if SERVAL one is not found or is nan.
    caracalin : str, astropy header object (astropy.io.fits.header.Header) or None (default)
        Three options:
            1) Path to directory containing CARACAL reduced FITS file (`filobs`) from which to read the header, or
            2) astropy header object (faster if header already loaded before), or
            ####3) None (default).
        If it is a header object, the parameter `ext` is not used.
        ####If None, try to get filename from SERVAL data.
    use_berv, use_drift, use_sa: bool, default True
        Consider or not the BERV, drift and secular acceleration corrections.
    """
    verboseprint = print if verb else lambda *a, **k: None

    timeidobs = filobs.replace('_A.fits', '.fits')

    header = False

    if use_caracal:
        # If `caracalin` is a str
        if isinstance(caracalin, str): caracal_filin = os.path.join(caracalin, filobs)
        else: caracal_filin = caracalin

    # Check if necessary SERVAL data already loaded
    if isinstance(servalin, pd.core.frame.DataFrame) and ('servalberv' in servalin) and ('servaldrift' in servalin) and ('servalsa' in servalin) and ('timeid' in servalin):
        dataserval = servalin
    # If not, load necessary SERVAL data
    else:
        dataserval = serval_get(servalin, obj=obj, lisdataid=['rvc', 'info'])
    # Make sure index is timeid, and not bjd
    if dataserval.index.name == 'bjd':
        dataserval['bjd'] = dataserval.index
        dataserval.set_index('timeid', inplace=True)

    # -----------------------------------------------------

    # BERV
    if use_berv:
        # Get BERV from SERVAL
        try:
            berv = dataserval['servalberv'].loc[timeidobs]
            berverr = np.nan
        except:
            berv = notfound
            berverr = notfound
        # If wrong BERV from SERVAL, get it from CARACAL
        if not np.isfinite(berv) and use_caracal:
            # Read header
            if isinstance(caracal_filin, str): header = fitsutils.read_header(caracal_filin, ext=ext)
            elif isinstance(caracal_filin, fits.header.Header): header = caracal_filin
            # Get BERV from header
            berv = caracal_berv(header, notfound=notfound)*1.e3  # [m/s]
            berverr = np.nan
    else:
        berv = np.nan
        berverr = np.nan

    # -----------------------------------------------------

    # Drift
    if use_drift:
        # Get drift from SERVAL
        try:
            drift = dataserval['servaldrift'].loc[timeidobs]
            drifterr = dataserval['servaldrifterr'].loc[timeidobs]
        except:
            drift = notfound
            drifterr = notfound
        # If wrong drift from SERVAL, get it from CARACAL
        if not np.isfinite(drift) and use_caracal:
            # Read header if not read before
            if header == False:
                if isinstance(caracal_filin, str): header = fitsutils.read_header(caracal_filin, ext=ext)
                elif isinstance(caracal_filin, fits.header.Header): header = caracal_filin
            # Get drift and drifterr from header
            drift, drifterr = caracal_drift(header, notfound=notfound, outfmt='single')  # [m/s]
    else:
        drift = np.nan
        drifterr = np.nan

    # -----------------------------------------------------

    # Secular acceleration
    if use_sa:
        # Get sa from SERVAL (no sa in FITS header)
        try:
            sa = dataserval['servalsa'].loc[timeidobs]
            saerr = np.nan
        except:
            sa = np.nan
            saerr = np.nan
    else:
        sa = np.nan
        saerr = np.nan

    # -----------------------------------------------------

    # Get shift
    dicshift = {'berv': berv, 'drift': drift, 'sa': sa}
    dicshift_correctsign = {'berv': -berv, 'drift': +drift, 'sa': +sa}
    dicshifterr = {'berverr': berverr, 'drifterr': drifterr, 'saerr': saerr}

    shift, shifterr = 0., 0.
    for k, v in dicshift_correctsign.items():
        if np.isfinite(v):
            shift += v
        else:
            verboseprint('Warning: {}={}. No {} correction applied!'.format(k, v, k))
    for k, v in dicshifterr.items():
        if np.isfinite(v):
            shifterr += v**2
        else:
            verboseprint('Warning: {}={}. No {} correction error applied!'.format(k, v, k))
    shifterr = np.sqrt(shifterr)

    # All data in dicshift
    dicshift.update(dicshifterr)  # Merge dictionaries
    dicshift['shift'] = shift
    dicshift['shifterr'] = shifterr

    return shift, shifterr, dicshift


def serval_rvcorrection_lisobs(servalin, obj=None, lisfilobs=None, servalnames=False, use_berv=True, use_drift=True, use_sa=True, join='outer'):
    """Get RV correction of all the observations from SERVAL data.

    See also `serval_caracal_rvcorrection_lisobs`. This function should be faster.

    If a value is a nan, it is considered a 0.

    To apply the correction:
        wcorrected = w * (1 - shift/c)

    Parameters
    ----------
    servalin : str or pd.DataFrame
        Path to directory containing SERVAL data or pandas dataframe with the necessary data (berv, drift and sa) already loaded.
    use_berv, use_drift, use_sa: bool, default True
        Consider or not the BERV, drift and secular acceleration corrections.
    lisfilobs : 1d array-like of str
        List with the (full paths to the) observations.
        Used to return only the observations listed here. If None (default), data for all the observations in the SERVAL outputs is returned.
    servalnames : bool
        Keep SERVAL names (`servalberv`) or remove 'serval' (`berv`)

    Returns
    -------
    shift : pandas dataframe
        Columns: timeid, shift
    shifterr : pandas dataframe
        Columns: timeid, shifterr
    dataused : pandas dataframe
        All data used to compute the shift and it error. Columns: timeid, RVs used, shift, shifterr
    """
    # Check if necessary SERVAL data already loaded
    if isinstance(servalin, pd.core.frame.DataFrame) and ('servalberv' in servalin.columns) and ('servaldrift' in servalin.columns) and ('servalsa' in servalin.columns) and ('timeid' in servalin.columns):
        dataserval = servalin
    # If not, load necessary SERVAL data
    else:
        dataserval = serval_get(servalin, obj=obj, lisdataid=['rvc', 'info'], join=join)

    # Make sure index is timeid, and not bjd
    if dataserval.index.name == 'bjd':
        dataserval['bjd'] = dataserval.index
        dataserval.set_index('timeid', inplace=True)

    # Select columns to be used
    cols_rv, cols_err = [], []
    if use_berv:
        cols_rv.extend(['servalberv'])
        cols_err.extend(['servalberverr'])
    if use_drift:
        cols_rv.extend(['servaldrift'])
        cols_err.extend(['servaldrifterr'])
    if use_sa:
        cols_rv.extend(['servalsa'])
        cols_err.extend(['servalsaerr'])
    cols = cols_rv + cols_err

    # Check all columns in dataframe. If not, create columns filled with nans
    for c in cols:
        if c not in dataserval.columns: dataserval[c] = np.nan

    # Transform nans into 0
    dataserval_nan0 = dataserval[cols].copy()
    dataserval_nan0.fillna(0, inplace=True)

    # Get shift: - BERV + drift + sa
    dataserval['shift'] = 0.
    for c in cols_rv:
        if c == 'servalberv':
            dataserval['shift'] += -1.*dataserval_nan0[c]
        else:
            dataserval['shift'] += dataserval_nan0[c]
    # Get shift err
    dataserval['shifterr'] = 0.
    for c in cols_err:
        dataserval['shifterr'] += dataserval_nan0[c]**2
    dataserval['shifterr'] = np.sqrt(dataserval['shifterr'])

    # Select observations to return
    if lisfilobs is not None:
        lisobs = [os.path.basename(o).replace('_A.fits', '.fits') for o in lisfilobs]
        dataserval = dataserval.loc[lisobs]

    # Output
    shift = dataserval['shift']
    shifterr = dataserval['shifterr']
    dataused = dataserval[cols+['shift', 'shifterr']]

    # Change column names
    if not servalnames:
        cols = {'servalberv': 'berv', 'servaldrift': 'drift', 'servalsa': 'sa', 'servalberverr': 'berverr', 'servaldrifterr': 'drifterr', 'servalsaerr': 'saerr'}
        dataused.rename(columns=cols, inplace=True)

    return shift, shifterr, dataused


def serval_caracal_rvcorrection_lisobs(servalin, obj=None, use_caracal=True, lisfilobs=None, servalnames=False, use_berv=True, use_drift=True, use_sa=True, notfound=np.nan, ext=0, verb=False):
    """Get RV correction of all the observations from SERVAL data and optionally, if not available, from CARACAL reduced FITS header.

    If a value is a nan, it is considered a 0.

    To apply the correction:
        wcorrected = w * (1 - shift/c)

    Parameters
    ----------
    servalin : str or pd.DataFrame
        Path to directory containing SERVAL data or pandas dataframe with the necessary data (berv, drift and sa) already loaded.
    use_berv, use_drift, use_sa: bool, default True
        Consider or not the BERV, drift and secular acceleration corrections.
    use_caracal : bool (default True)
        Use value from CARACAL header if SERVAL one is not found or is nan. If True, must provide the path to the observations with `lisfilobs`.
    lisfilobs : 1d array-like of str
        List with the (full paths to the) observations.
        If `use_caracal` is True, must provide the full paths to the observations.
        If `use_caracal` is False, `lisfilobs` does not need to be the full path, but only the list of observations. It is used to return only the observations listed here. If None (default), data for all the observations in the SERVAL outputs is returned.
    servalnames : bool
        Keep SERVAL names (`servalberv`) or remove 'serval' (`berv`)

    Returns
    -------
    shift : pandas dataframe
        Columns: timeid, shift
    shifterr : pandas dataframe
        Columns: timeid, shifterr
    dataused : pandas dataframe
        All data used to compute the shift and it error. Columns: timeid, RVs used, shift, shifterr
    """

    # Only use SERVAL data
    if not use_caracal:

        # Check if necessary SERVAL data already loaded
        if isinstance(servalin, pd.core.frame.DataFrame) and ('servalberv' in servalin.columns) and ('servaldrift' in servalin.columns) and ('servalsa' in servalin.columns) and ('timeid' in servalin.columns):
            dataserval = servalin
        # If not, load necessary SERVAL data
        else:
            dataserval = serval_get(servalin, obj=obj, lisdataid=['rvc', 'info'])

        # Make sure index is timeid, and not bjd
        if dataserval.index.name == 'bjd':
            dataserval['bjd'] = dataserval.index
            dataserval.set_index('timeid', inplace=True)

        # Select columns to be used
        cols_rv, cols_err = [], []
        if use_berv:
            cols_rv.extend(['servalberv'])
            cols_err.extend(['servalberverr'])
        if use_drift:
            cols_rv.extend(['servaldrift'])
            cols_err.extend(['servaldrifterr'])
        if use_sa:
            cols_rv.extend(['servalsa'])
            cols_err.extend(['servalsaerr'])
        cols = cols_rv + cols_err

        # Check all columns in dataframe. If not, create columns filled with nans
        for c in cols:
            if c not in dataserval.columns: dataserval[c] = np.nan

        # Transform nans into 0
        dataserval_nan0 = dataserval[cols].copy()
        dataserval_nan0.fillna(0, inplace=True)

        # Get shift: - BERV + drift + sa
        dataserval['shift'] = 0.
        for c in cols_rv:
            if c == 'servalberv':
                dataserval['shift'] += -1.*dataserval_nan0[c]
            else:
                dataserval['shift'] += dataserval_nan0[c]
        # Get shift err
        dataserval['shifterr'] = 0.
        for c in cols_err:
            dataserval['shifterr'] += dataserval_nan0[c]**2
        dataserval['shifterr'] = np.sqrt(dataserval['shifterr'])

        # Select observations to return
        if lisfilobs is not None:
            lisobs = [os.path.basename(o).replace('_A.fits', '.fits') for o in lisfilobs]
            dataserval = dataserval.loc[lisobs]

        # Output
        shift = dataserval['shift']
        shifterr = dataserval['shifterr']
        dataused = dataserval[cols+['shift', 'shifterr']]

        # Change column names
        if not servalnames:
            cols = {'servalberv': 'berv', 'servaldrift': 'drift', 'servalsa': 'sa', 'servalberverr': 'berverr', 'servaldrifterr': 'drifterr', 'servalsaerr': 'saerr'}
            dataused.rename(columns=cols, inplace=True)

    # -----------------------

    # Use SERVAL data and if not available or nan, use data from CARACAL FITS header
    else:
        dataused = {}
        for obs in lisfilobs:
            filobs = os.path.basename(obs)
            dirobs = os.path.dirname(obs)
            _, _, dicshift_o = serval_caracal_rvcorrection(filobs, servalin, obj=obj, use_caracal=use_caracal, caracalin=dirobs, use_berv=use_berv, use_drift=use_drift, use_sa=use_sa, notfound=notfound, ext=ext, verb=verb)
            timeid = filobs.replace('_A.fits', '.fits')
            dataused[timeid] = dicshift_o

        # Outpus
        dataused = pd.DataFrame.from_dict(dataused, orient='index')
        shift = dataused['shift']
        shifterr = dataused['shifterr']

    return shift, shifterr, dataused

###############################################################################

# -----------------------------------------------------------------------------
#
# Nightly zero points utils (zero)
#
# -----------------------------------------------------------------------------


def zero_read_avc(fil, ifnofilout='empty', nrow=1):
    """
    SI : bool, default True
        Return data in the international system of units (m/s instead of km/s, etc)

    Description of file: obj.avc.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       AVC       Radial velocity (drift and NZP corrected)
         3 D      m/s     E_AVC       Radial velocity error
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'zeroavc', 'zeroavcerr']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def zero_read_avcn(fil, SI=True, ifnofilout='empty', nrow=1):
    """
    SI : bool, default True
        Return data in the international system of units (m/s instead of km/s, etc)

    Description of file: obj.avcn.dat
    --------------------------------------------------------------------------------
    Column Format Units     Label     Explanations
    --------------------------------------------------------------------------------
         1 D      ---       BJD       Barycentric Julian date [1]
         2 D      m/s       AVC       Radial velocity (drift and NZP corrected)
         3 D      m/s     E_AVC       Radial velocity error
         4 D      m/s       DRIFT     CARACAL drift measure
         5 D      m/s     E_DRIFT     CARACAL drift measure error
         6 D      m/s       RV        Radial velocity
         7 D      m/s     E_RV        Radial velocity error
         8 D      km/s      BERV      Barycentric earth radial velocity [1]
         9 D      m/s       SADRIFT   Drift due to secular acceleration
        10 D      m/s       NZP       Nightly zero point (NZP)
        11 D      m/s     E_NZP       Error for NZP
        12 I      ---       FLAG_AVC  Byte flag: 0 (ok), 1 (NZP replaced by median NZP), 2 (nan drift)
    --------------------------------------------------------------------------------
    """
    column_names = ['bjd', 'zeroavc', 'zeroavcerr', 'zerodrift', 'zerodrifterr', 'zerorv', 'zerorverr', 'zeroberv', 'zerosa', 'zeronzp', 'zeronzperr', 'zeroavcflag']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    if SI:
        data['zeroberv'] = data['zeroberv'] * 1.e3  # [m/s]
    return data


###############################################################################

# -----------------------------------------------------------------------------
#
# CCF utils
#
# -----------------------------------------------------------------------------


def ccf_read_par(fil, SI=True):
    """
    obj.par.dat (all data)

    SI : bool, default True
        Return the RV and BIS in m/s and not km/s. The FWHM is still returned in km/s!
    """
    # data = pd.read_csv(fil, sep=' ')
    try:
        data = pd.read_csv(fil, sep='\s+', header=0, index_col=0)
        if SI:
            data['rv'] = data['rv'] * 1.e3  # [m/s]
            data['rverr'] = data['rverr'] * 1.e3  # [m/s]
            data['bis'] = data['bis'] * 1.e3  # [m/s]
            data['biserr'] = data['biserr'] * 1.e3  # [m/s]
    # If file does not exist
    except:
        # The real columns in the par.dat file may change
        column_names = ['obs', 'fitshift', 'fitshifterr', 'fitwid', 'fitwiderr', 'fitcen', 'fitcenerr', 'fitamp', 'fitamperr', 'fitredchi2', 'rv', 'rverr', 'rverrabs', 'fwhm', 'fwhmerr', 'contrast', 'contrasterr', 'bis', 'biserr', 'berv', 'drift', 'sa', 'berverr', 'drifterr', 'saerr', 'shift', 'shifterr', 'otherdrift', 'otherdrifterr', 'bjd', 'oref', 'snroref', 'ron', 'exptime', 'filmask', 'filmaskname', 'objmask', 'sptmask', 'vsinimask', 'nlino0', 'nlino1', 'nlino2', 'nlino3', 'nlino4', 'nlino5', 'nlino6', 'nlino7', 'nlino8', 'nlino9', 'nlino10', 'nlino11', 'nlino12', 'nlino13', 'nlino14', 'nlino15', 'nlino16', 'nlino17', 'nlino18', 'nlino19', 'nlino20', 'nlino21', 'nlino22', 'nlino23', 'nlino24', 'nlino25', 'nlino26', 'nlino27', 'nlino28', 'nlino29', 'nlino30', 'nlino31', 'nlino32', 'nlino33', 'nlino34', 'nlino35', 'nlino36', 'nlino37', 'nlino38', 'nlino39', 'nlino40', 'nlino41', 'nlino42', 'nlino43', 'nlino44', 'nlino45', 'nlino46', 'nlino47', 'nlino48', 'nlino49', 'nlino50', 'nlino51', 'nlino52', 'nlino53', 'nlino54', 'nlino55', 'nlino56', 'nlino57', 'nlino58', 'nlino59', 'nlino60', 'nlint', 'nlintallords', 'nlinoriginal', 'filobs']
        data = pd.DataFrame(columns=column_names)
        data.set_index(column_names[0], inplace=True)
    return data


def ccf_read_ccfpar(fil, SI=True, ifnofilout='empty', nrow=1):
    """
    obj.ccfpar.dat (only CCF parameters)

    SI : bool, default True
        Return the RV and BIS in m/s and not km/s. The FWHM is still returned in km/s!
    """
    column_names = ['bjd', 'ccfrv', 'ccffwhm', 'ccfcontrast', 'ccfbis', 'ccfrverr', 'ccffwhmerr', 'ccfcontrasterr', 'ccfbiserr']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    if SI:
        data['ccfrv'] = data['ccfrv'] * 1.e3  # [m/s]
        data['ccfrverr'] = data['ccfrverr'] * 1.e3  # [m/s]
        data['ccfbis'] = data['ccfbis'] * 1.e3  # [m/s]
        data['ccfbiserr'] = data['ccfbiserr'] * 1.e3  # [m/s]
    return data


###############################################################################

# -----------------------------------------------------------------------------
#
# pEW utils
#
# -----------------------------------------------------------------------------

def pew_read_pewvis(fil, ifnofilout='empty', nrow=1):
    """
    obj.pEW.dat: time series of pEW' values
    Description of file: obj.pEW.dat
    -----------------------------------------------------------------------------
    Col Format Units  Label             Explanations
    -----------------------------------------------------------------------------
      1 D      d      BJD               Barycentric Julian date (from SERVAL)
      2 D      Å      Halpha            H alpha pEW'
      3 D      Å      Halpha_err        H alpha pEW' error
      4 D      ---    log_L_Halpha      normalized luminosity log(L_Halpha/L_bol)
      5 D      ---    log_L_Halpha_err  normalized luminosity error
      6 D      Å      HeD3              He D3 pEW'
      7 D      Å      HeD3_err          He D3 pEW' error
      8 D      Å      NaD2              Na D2 pEW'
      9 D      Å      NaD2_err          Na D2 pEW' error
     10 D      Å      NaD1              Na D1 pEW'
     11 D      Å      NaD1_err          Na D1 pEW' error
     12 D      Å      NaD               Na D pEW' (Na D1 + Na D2)
     13 D      Å      NaD_err           Na D pEW' error
     14 D      Å      CaIRTa            CaII-IRT-a pEW'
     15 D      Å      CaIRTa_err        CaII-IRT-a pEW' error
     16 D      Å      CaIRTb            CaII-IRT-b pEW'
     17 D      Å      CaIRTb_err        CaII-IRT-b pEW' error
     18 D      Å      CaIRTc            CaII-IRT-c pEW'
     19 D      Å      CaIRTc_err        CaII-IRT-c pEW' error
     20 D      Å      Fe8691            Fe 8691Å pEW'
     21 D      Å      Fe8691_err        Fe 8691Å pEW' error
    -----------------------------------------------------------------------------
    """
    column_names = ['bjd', 'halpha', 'halphaerr', 'loglhalpha', 'loglhalphaerr', 'hed3', 'hed3err', 'nad2', 'nad2err', 'nad1', 'nad1err', 'nad', 'naderr', 'cairta', 'cairtaerr', 'cairtb', 'cairtberr', 'cairtc', 'cairtcerr', 'fe8691', 'fe8691err']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def pew_read_cahband(fil, ifnofilout='empty', nrow=1):
    """
    obj.CaH.band.dat: time series of CaH indices
    Description of file: obj.CaH.band.dat
    -------------------------------------------------------------------------
    Col Format Units  Label             Explanations
    -------------------------------------------------------------------------
      1 D      d      BJD               Barycentric Julian date (from SERVAL)
      2 D      ---    CaH2              CaH2 index
      3 D      ---    CaH2_err          CaH2 index error
      4 D      ---    CaH3              CaH3 index
      5 D      ---    CaH3_err          CaH3 index error
    -------------------------------------------------------------------------
    """
    column_names = ['bjd', 'cah2', 'cah2err', 'cah3' 'cah3err']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def pew_read_tioband(fil, ifnofilout='empty', nrow=1):
    """
    obj.TiO.band.dat: time series of TiO indices
    Description of file: obj.TiO.band.dat
    -------------------------------------------------------------------------
    Col Format Units  Label             Explanations
    -------------------------------------------------------------------------
      1 D      d      BJD               Barycentric Julian date (from SERVAL)
      2 D      ---    TiO7050           TiO 7050 index
      3 D      ---    TiO7050_err       TiO 7050 index error
      4 D      ---    TiO8430           TiO 8430 index
      5 D      ---    TiO8430_err       TiO 8430 index error
      6 D      ---    TiO8860           TiO 8860 index
      7 D      ---    TiO8860_err       TiO 8860 index error
    -------------------------------------------------------------------------
    """
    column_names = ['bjd', 'tio7050', 'tio7050err', 'tio8430' 'tio8430err', 'tio8860', 'tio8860err']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def pew_read_voband(fil, ifnofilout='empty', nrow=1):
    """
    obj.VO.band.dat: time series of VO indices
    Description of file: obj.VO.band.dat
    -------------------------------------------------------------------------
    Col Format Units  Label             Explanations
    -------------------------------------------------------------------------
      1 D      d      BJD               Barycentric Julian date (from SERVAL)
      2 D      ---    VO7436            VO 7436 index
      3 D      ---    VO7436_err        VO 7436 index error
      4 D      ---    VO7942            VO 7942 index
      5 D      ---    VO7942_err        VO 7942 index error
    -------------------------------------------------------------------------
    """
    column_names = ['bjd', 'vo7436', 'vo7436err', 'vo7942' 'vo7942err']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def pew_read_pewnir(fil, ifnofilout='empty', nrow=1):
    """
    obj.pEW.dat: time series of pEW' values
    Description of file: obj.pEW.dat
    --------------------------------------------------------------------------------
    Col Format Units  Label             Explanations
    --------------------------------------------------------------------------------
      1 D      d      BJD               Barycentric Julian date (from SERVAL)
      2 D      Å      He10833           He 10833 pEW'
      3 D      Å      He10833_err       He 10833 pEW' error
      4 I      ---    He10833_telluric  1 if He 10833 window contains telluric lines
      5 D      Å      Pabeta            Paschen beta pEW'
      6 D      Å      Pabeta_err        Paschen beta pEW' error
      7 I      ---    Pabeta_telluric   1 if Pa beta window contains telluric lines
    --------------------------------------------------------------------------------
    He 10833 is marked as contaminated by tellurics if the absolute Doppler shift of the stellar spectrum is between 14 and 125 km/s.
    Pa beta is marked as contaminated by tellurics if the absolute Doppler shift of the stellar spectrum is between 10 and 43 km/s.
    """
    column_names = ['bjd', 'he10833', 'he10833err', 'he10833telluric' 'pabeta', 'pabetaerr', 'pabetatelluric']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


def pew_read_wfb(fil, ifnofilout='empty', nrow=1):
    """
    obj.WFB.band.dat: time series of FeH Wing-Ford band (WFB) index
    Description of file: obj.WFB.band.dat
    -------------------------------------------------------------------------
    Col Format Units  Label             Explanations
    -------------------------------------------------------------------------
      1 D      d      BJD               Barycentric Julian date (from SERVAL)
      2 D      ---    WFB               FeH Wing-Ford band index
      3 D      ---    WFB_err           FeH Wing-Ford band index error
    -------------------------------------------------------------------------
    """
    column_names = ['bjd', 'wfb', 'wfberr']
    data = read_file2dataframe(fil, column_names, ifnofilout=ifnofilout, nrow=nrow)
    return data


###############################################################################

# -----------------------------------------------------------------------------
#
# CARMENCITA utils
#
# -----------------------------------------------------------------------------

def carmencita_version_latest(dirin, verbose=False):
    """Find Carmencita latest version in `dirin`.

    Parameters
    ----------
    dirin : str
        Directory containing the Carmencita(s) database(s). Relative to home directory.

    Returns
    -------
    fil : str
        Complete filename of the latest Carmencita version.
    """
    verboseprint = print if verbose else lambda *a, **k: None

    version_code = '[0-9]'*3  # 3 integers
    lisfil = np.sort(glob.glob(os.path.join(dirin, 'carmencita.{}.csv'.format(version_code))))
    fil = lisfil[-1]

    verboseprint('{} Carmencita versions available in {}:'.format(len(lisfil), dirin))
    for i in lisfil: verboseprint('', os.path.basename(i))  # Print all the files found
    verboseprint('File selected (latest version): {}'.format(fil))
    return fil


def carmencita_filename_version(dirin, version=None, verbose=False):
    """Get Carmencita filename depending on version specified: a specific number or the latest one.

    Parameters
    ----------
    dirin : str
        Directory containing the Carmencita(s) database(s). Relative to home directory.
    version : int or None
        Carmencita version (i.e. an integer) or None. If None, will look for the latest version, i.e. the one with the largest integer in the filename.

    Returns
    -------
    fil : str
        Complete filename of the Carmencita version specified.
    """
    verboseprint = print if verbose else lambda *a, **k: None

    if version is None:
        verboseprint('Looking for Carmencita latest version')
        fil = carmencita_version_latest(dirin, verbose)
    else:
        verboseprint('Looking for Carmencita version {}'.format(version))
        version_code = '{:03d}'.format(version)
        fil = os.path.join(dirin, 'carmencita.{}.csv'.format(version_code))
    return fil


def carmencita_read(filin, verbose=False):
    """Read Carmencita database specified in `filin` into a pandas dataframe.

    Parameters
    ----------
    filin : str
        Complete path to file.

    Returns
    -------
    data : pandas dataframe
        Carmencita data read from `filin`.
    """
    if verbose: print('Reading Carmencita file: {}'.format(filin))
    try:
        data = pd.read_csv(filin, sep=',', header=0, names=None, index_col=0, usecols=None)
    except IOError as err:
        print('{}: {}\nCheck directory and/or Carmencita version number given. Exit.'.format(type(err).__name__, err, filin))
        print(err.args)
        sys.exit(1)
        # sys.exit('{}\nCannot read file {}. Check if it exists (check directory and/or version number).'.format(err, filin))
    return data


def carmencita_get(dirin, version=None, verbose=False):
    """Load Carmencita. Can select input directory and version.

    Parameters
    ----------
    dirin : str
        Directory containing the Carmencita(s) database(s). Relative to home directory.
    version : int or None
        Carmencita version (i.e. an integer) or None. If None, will look for the latest version, i.e. the one with the largest integer in the filename.

    Returns
    -------
    data : pandas dataframe
        Carmencita data.
    """
    # Get filename depending on version specified (a specific one or the latest one)
    filcarmencita = carmencita_filename_version(dirin, version=version, verbose=verbose)

    # Read Carmencita into a pandas dataframe
    data = carmencita_read(filcarmencita, verbose=verbose)

    return data
