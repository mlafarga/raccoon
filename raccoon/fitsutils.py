#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import sys

from astropy.io import fits
import numpy as np
import pandas as pd

###############################################################################


# Read header

def read_header(filin, ext=0):
    """
    Read FITS header. Can choose which extension with `ext` (default 0).
    """
    with fits.open(filin) as hdulist:
        header = hdulist[ext].header
    return header


def read_header_keywords(filin, kws, notfound=np.nan, ext=0, names=None):
    """Read keywords in `kws` from FITS header.

    Parameters
    ----------
    filin : str or astropy header object (astropy.io.fits.header.Header)
        FITS file from which to read the header or astropy header object (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    kws : str or list-like
    notfound : any (default np.nan)
        Value to write in the output if the keyword is not found in the header.
    ext : int, str (default 0)
        Extension of the FITS from which to get the header.
    names : dict
        Names for the output dictionary keys, to substitute the original header keywords. E.g. `names={'oldk: 'newk}`.
        Must contain all the keywords in `kws`.
        If not a dictionary or old keywords not correct, no change is made.

    Returns
    -------
    data : dict
    """

    # Read header
    if isinstance(filin, str): header = read_header(filin, ext=ext)
    elif isinstance(filin, fits.header.Header): header = filin

    # Make sure kws is a list-type
    if not isinstance(kws, (list, tuple, np.ndarray)): kws = [kws]

    # Get keywords values
    data = {}
    for kw in kws:
        try: data[kw] = header[kw]
        except: data[kw] = notfound

    # Change dict keys
    # Make sure names is a dict
    if isinstance(names, dict):
        # Make sure all old kws in names
        if data.keys() == names.keys():
            # Change kws
            datanew = {names[k]: v for k, v in data.items()}
            data = datanew

    return data


def read_header_keywords_lisobs(lisobs, kws, notfound=np.nan, ext=0, names=None):
    """Read keywords in `kws` from FITS header for the observations in `lisobs`.

    Parameters
    ----------
    lisobs : list of str or list of astropy header object (astropy.io.fits.header.Header)
        List of the FITS files from which to read the header or list of astropy header objects (faster if header already loaded before). If it is a header object, the parameter `ext` is not used.
    kws : str or list-like
    notfound : any (default np.nan)
        Value to write in the output if the keyword is not found in the header.
    ext : int, str (default 0)
        Extension of the FITS from which to get the header.
    names : dict
        Names for the output dictionary keys, to substitute the original header keywords. E.g. `names={'oldk: 'newk}`.
        Must contain all the keywords in `kws`.
        If not a dictionary or old keywords not correct, no change is made.

    Returns
    -------
    data : pandas dataframe
        Index: lisobs, Columns: kws
    """

    # Make sure kws is a list-type
    if not isinstance(kws, (list, tuple, np.ndarray)): kws = [kws]

    lisdata = []
    for obs in lisobs:
        # Read header
        if isinstance(obs, str): header = read_header(obs, ext=ext)
        elif isinstance(obs, fits.header.Header): header = obs

        # Get keywords values
        dataobs = {}
        for kw in kws:
            try: dataobs[kw] = header[kw]
            except: dataobs[kw] = notfound

        lisdata.append(dataobs)

    # Convert to dataframe
    data = pd.DataFrame(lisdata, index=lisobs)

    # Change dataframe columns
    if isinstance(names, dict):
        if list(data.columns) == list(names.keys()):
            data.rename(columns=names, inplace=True)

    return data


# Output header

def list_to_header(datain):
    """
    datain : list or tuple of tuples
        Input data for the header.
        Each tuple must have at least a keyword (1 entry minimum). The value (second entry)  and the comment (third entry) are optional (3 entries maximum). If no value is provided, it defaults to and empty string ''.
        E.g.
            datain = (('key1', 3, 'comment1'), ('key2', 'val2'), ('key3'))
        or
            datain = [('key1', 3, 'comment1'), ('key2', 'val2'), ('key3')]
    """
    # Check input format ok
    if not isinstance(datain, (list, tuple)):
        sys.exit('Data input type not correct.\nType: {}\nData: {}'.format(type(datain), datain))
    # Make header
    header = fits.Header(cards=datain)
    return header


def add_list_to_header(datain, headerin):
    """
    Uses astropy HEADER.extend.
    """
    # Check input format ok
    if not isinstance(datain, (list, tuple)):
        sys.exit('Data input type not correct.\n{}'.format(datain))
    if not isinstance(headerin, fits.header.Header):
        sys.exit('Header input type not correct.\n{}'.format(headerin))
    # Add data to header
    headerout = headerin.copy()  # Do not modify input header
    headerout.extend(datain)
    return headerout
