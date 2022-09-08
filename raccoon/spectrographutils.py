#!/usr/bin/env python
"""
Tools to work with data from different spectrographs: CARMENES, HARPS, HARPN.
Uses functions from `harpsutils` and `carmenesutils`.
"""
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd

from . import ccf as ccflib
from . import carmenesutils
from . import harpsutils
from . import espressoutils
from . import expresutils

###############################################################################

# Spectrograph data
# -----------------

# Pixels per order
dicpix = {
    'CARM_VIS': 4096,
    'ESPRESSO': 9111,
}

# Resolution
dicres = {
    'CARM_VIS': 94600,
    'CARM_NIR': 80400,
    'HARPS': 115000,
    'HARPN': 115000,
    'EXPRES': 150000,
    'ESPRESSO': 140000, # If HR mode (1-UT). Can also be 190000 if UHR mode (1-UT) or 70000 if MR mode (4-UT)
}

# Number of orders
dicnord = {
    'CARM_VIS': 61,
    'CARM_NIR': 28,
    'HARPS': 72,
    'HARPN': 69,
    'EXPRES': 86,
    'ESPRESSO': 170,
}


def inst_nord(inst, carmnirsplit=True, notfound=None, verb=True):
    """Get number of orders for instrument `inst`.

    Parameters
    ----------
    carmnirsplit : bool (default True)
        Multiply CARM_NIR orders by 2. Usually use CARM_NIR orders split by half, so have double of orders.

    Returns
    -------
    nord : int
    """
    try:
        nord = dicnord[inst]
        if inst == 'CARM_NIR' and carmnirsplit:
            nord += nord  # double
    except:
        if verb: print('Instrument {} not available, return {}'.format(inst, notfound))
        nord = notfound
    return nord


# Reference order

dicoref = {
   'CARM_VIS': 36,
   'CARM_NIR': 11,  # This is the double already
   'HARPS': 55,
   'HARPN': 55,
   'EXPRES': 60,  # ?
   # 'ESPRESSO': 106,  # ? Orders 106 and 107 (starting from 0) cover the 550 nm (orders 104 and 105 too, but 550 nnm falls on the extreme)
   'ESPRESSO': 102,  # ? Orders 102 and 103 (starting from 0) cover the 550 nm (orders 100 and 101 too, but 550 nnm falls on the extreme)
}


def inst_oref(inst, carmnirsplit=True, notfound=None, verb=True):
    """Get reference order for instrument `inst`.

    Parameters
    ----------
    carmnirsplit : bool (default True)
        Multiply CARM_NIR `oref` by 2. Usually use CARM_NIR orders split by half, so have double of orders.

    Returns
    -------
    oref : int
    """
    try:
        oref = dicoref[inst]
        if inst == 'CARM_NIR' and carmnirsplit == False:
            oref = int(oref / 2)  # half
    except:
        if verb: print('Instrument {} not available, return {}'.format(inst, notfound))
        oref = notfound
    return oref


# RV per pixel [km/s]

dicrvpixmed = {
    'CARM_VIS': 1.258,
    'CARM_NIR': 1.356,
    'HARPS': 0.820,
    'HARPN': 0.820,
    'EXPRES': 0.500,
}


def inst_rvpixmed(inst, notfound=None, verb=True):
    """Get the median delta RV per pixel for instrument `inst`.

    Parameters
    ----------

    Returns
    -------
    rvpixmed : int
    """
    try:
        rvpixmed = dicrvpixmed[inst]
    except:
        if verb: print('Instrument {} not available, return {}'.format(inst, notfound))
        rvpixmed = notfound
    return rvpixmed


# Spectral sampling s [pix/SE] (SE: spectral element, ~ FWHM)

dictspix = {
    'CARM_VIS': 2.5,
    'CARM_NIR': 2.8,
    'HARPS': 3.2,
    'HARPN': 3.2,
    'EXPRES': 3.6,  # 4.0
}


def inst_rvpixmed(inst, notfound=None, verb=True):
    """Get sampling [pix/SE] for instrument `inst`.

    Parameters
    ----------

    Returns
    -------
    rvpixmed : int
    """
    try:
        rvpixmed = dictspix[inst]
    except:
        if verb: print('Instrument {} not available, return {}'.format(inst, notfound))
        rvpixmed = notfound
    return rvpixmed

###############################################################################


# Reduced spectra
# ---------------

# Read reduced spectrum
def fitsred_read(filin, inst, 
    # CARMENES
    carmnirdiv=True, 
    # HARPS/N
    harpblaze=True, dirblaze=None, filblaze=None,
    # EXPRES
    expresw='bary_excalibur',
    ):
    """
    Parameters
    ----------
    carmnirdiv : bool, default True
        If True, divide the orders by the discontinuity at the center. If not, leave them as read from the FTIS. Only works for `inst='CARM_NIR'`.
    harpblaze : bool, default True
        If True, get the blaze function from the corresponding file (see `dirblaze` and `filblaze`). If False, the output corresponding to the blaze, `c`, is an array of ones lwith the shape of `f`.
    dirblaze : str, default None
        Directory containing the blaze file. If None (default), assume the blaze file is in the same directory as `filin`.
    harpfilbalze : str, default None
        Blaze file name. If None (default), get the file name from the header.

    Returns
    -------
    """
    if inst == 'CARM_VIS':
        w, f, sf, c, header = carmenesutils.caracal_fitsred_read(filin)
        dataextra = {}
    elif inst == 'CARM_NIR':
        w, f, sf, c, header = carmenesutils.caracal_fitsred_read(filin)
        dataextra = {}
        if carmnirdiv:
            # w, f, sf, c = carmenesutils.caracal_fitsrednir_divide_ords(w=w, f=f, sf=sf, c=c)
            a = carmenesutils.caracal_fitsrednir_divide_ords(w=w, f=f, sf=sf, c=c)
            w, f, sf, c = a['w'], a['f'], a['sf'], a['c']

    elif inst == 'HARPS' or inst == 'HARPN':
        w, f, c, header, _ = harpsutils.drs_e2dsred_read(filin, readblaze=harpblaze, dirblaze=dirblaze, filblaze=filblaze, inst=inst)
        sf = np.zeros_like(w)
        dataextra = {}

    elif inst == 'EXPRES':
        w, wcb, we, wecb, mwecb, f, sf, c, b, mf, header, header1, header2 = expresutils.drs_fitsred_read(filin)
        if expresw == 'bary_excalibur':
            w = wecb
        elif expresw == 'excalibur':
            w = we
        elif expresw == 'bary_wavelength':
            w = w
        elif expresw == 'wavelength':
            w = wcb
        dataextra = {'blaze': b, 'pixel_mask': mf, 'excalibur_mask': mwecb, 'header1': header1, 'header2': header2}

    return w, f, sf, c, header, dataextra

# -----------------------------------------------------------------------------


# Values from header
# ------------------

# Get BJD from header
def header_bjd_lisobs(lisobs, inst, name='bjd', notfound=np.nan, ext=0):
    """
    Get the BJD from the header of the observations in `lisobs`.

    Parameters
    ----------
    name : str or None (default 'bjd')
        Change the pandas dataframe column name to `name`. If `None`, keep the header keyword as the column name.
    """
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        lisbjd = carmenesutils.caracal_bjd_lisobs(lisobs, notfound=notfound, ext=ext, name=name)
        # # Change column names
        # if name is not None:
        #     lisbjd.rename(columns={'HIERARCH CARACAL BJD': name}, inplace=True)
    elif inst == 'HARPS' or inst == 'HARPN':
        lisbjd = harpsutils.drs_bjd_lisobs(lisobs, inst, notfound=notfound, ext=ext, name=name)
    elif inst == 'ESPRESSO':
        lisbjd = espressoutils.drs_bjd_lisobs(lisobs, notfound=notfound, ext=ext, name=name)
    elif inst == 'EXPRES':
        lisbjd = expresutils.drs_bjd_lisobs(lisobs, notfound=notfound, ext=ext, name=name)
    return lisbjd


# Get readout noise RON from header
def header_ron_lisobs(lisobs, inst, name='ron', notfound=np.nan, ext=0):
    """
    Get the RON from the header of the observations in `lisobs`.

    Parameters
    ----------
    name : str or None (default 'ron')
        Change the pandas dataframe column name to `colname`. If `None`, keep the header keyword as the column name.
    """
    if inst == 'CARM_VIS':
        lisron = carmenesutils.caracal_ron_lisobs_vis(lisobs, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            lisron.rename(columns={'E_READN1': name}, inplace=True)

    elif inst == 'CARM_NIR':
        lisron = carmenesutils.caracal_ron_lisobs_nir(lisobs, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            lisron.rename(columns={'E_READN': name}, inplace=True)

    elif inst == 'HARPS' or inst == 'HARPN':
        lisron = harpsutils.drs_ron_lisobs(lisobs, inst, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            kwinst = harpsutils.headerkwinst(inst, outfail=np.nan)
            lisron.rename(columns={kwinst + 'DRS CCD SIGDET': name}, inplace=True)

    elif inst == 'EXPRES':
        # TODO: set to 0 for now
        lisron = pd.DataFrame(np.zeros_like(lisobs, dtype=float), columns=[name], index=lisobs)

    return lisron


# Get exposure time from header
def header_exptime_lisobs(lisobs, inst, name='exptime', notfound=np.nan, ext=0):
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        lisexptime = carmenesutils.caracal_exptime_lisobs(lisobs, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            lisexptime.rename(columns={'EXPTIME': name}, inplace=True)

    elif inst == 'HARPS' or inst == 'HARPN':
        lisexptime = harpsutils.drs_exptime_lisobs(lisobs, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            lisexptime.rename(columns={'EXPTIME': name}, inplace=True)

    if inst == 'EXPRES':
        lisexptime = expresutils.drs_exptime_lisobs(lisobs, notfound=notfound, ext=ext, name=name)

    return lisexptime


# Get airmass time from header
def header_airmass_lisobs(lisobs, inst, name='airmass', notfound=np.nan, ext=0):
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        lisairmass = carmenesutils.caracal_airmass_lisobs(lisobs, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            lisairmass.rename(columns={'AIRMASS': name}, inplace=True)

    elif inst == 'HARPS' or inst == 'HARPN':
        lisairmass = harpsutils.drs_airmass_lisobs(lisobs, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            lisairmass.rename(columns={'AIRMASS': name}, inplace=True)

    elif inst == 'EXPRES':
        lisairmass = expresutils.drs_airmass_lisobs(lisobs, notfound=notfound, ext=ext, name=name)

    return lisairmass


# Get SNR from header
def header_snr_lisobs(lisobs, inst, name='snro', ords=None, notfound=np.nan, ext=0):
    """
    Get the SNR from the header of the orders `ords` for the observations in `lisobs`.

    Parameters
    ----------
    name : {'ord', 'snro'} or None
        Change to pandas dataframe column name. If `ord`, change to the order number (an int, e.g. 36). If `snro`, change to e.g. `snro36`. If None, keep the header keyword as the column name.
    """
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        lissnr = carmenesutils.caracal_snr_lisobs(lisobs, ords=ords, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            if name == 'ord':
                changecol = {i: int(i.replace('HIERARCH CARACAL FOX SNR ', '')) for i in lissnr.columns}
            elif name == 'snro':
                changecol = {i: i.replace('HIERARCH CARACAL FOX SNR ', 'snro') for i in lissnr.columns}
            lissnr.rename(columns=changecol, inplace=True)

    elif inst == 'HARPS' or inst == 'HARPN':
        lissnr = harpsutils.drs_snr_lisobs(lisobs, ords=ords, notfound=notfound, ext=ext)
        # Change column names
        if name is not None:
            kwinst = harpsutils.headerkwinst(inst, outfail=np.nan)
            if name == 'ord':
                changecol = {i: int(i.replace('{}DRS SPE EXT SN'.format(kwinst), '')) for i in lissnr.columns}
            elif name == 'snro':
                changecol = {i: i.replace('{}DRS SPE EXT SN'.format(kwinst), 'snro') for i in lissnr.columns}
            lissnr.rename(columns=changecol, inplace=True)

    elif inst == 'EXPRES':
        # EXPRES S/N not in FITS header, get from spectrum
        lissnr = expresutils.drs_snr_lisobs(lisobs, ords, name=name)

    return lissnr


# Get RV corrections from header
def header_rvcorrection_lisobs(lisobs, inst, name='shift', notfound=np.nan, ext=0):
    """
    Get RV correction from header: BERV and drift.
    No secular acceleration or nightly drifts.

    name : str
        Change original header keyword to `name`. Not implemented in `carmenesutils.caracal_rvcorrection_lisobs` yet
    """
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        shift, shifterr, datashift = carmenesutils.caracal_rvcorrection_lisobs(lisobs, use_berv=True, use_drift=True, notfound=notfound)
    elif inst == 'HARPS' or inst == 'HARPN':
        shift, shifterr, datashift = harpsutils.drs_rvcorrection_lisobs(lisobs, inst, name=name, notfound=notfound, ext=ext)
    return shift, shifterr, datashift


# Get RV corrections from SERVAL or if not, from header
def serval_header_rvcorrection_lisobs(lisobs, inst, source='header', servalin=None, obj=None, notfound=np.nan, ext=0, join='outer'):
    """
    Parameters
    ----------
    source : {'header', 'serval', 'serval_header', 'none'} or filename
        Source from which to get the RV correction.
        - `header`: get it from FITS header. Can only get BERV and drift.
        - `serval`: get it from SERVAl outputs. Get BERV, drift and sa.
        - `serval_header`: try to get it from SERVAL, if not possible or nan, get it from header.
        If `serval` or `serval_header`, must provide `servalin`.
        - `none`: rv corrections are 0.
        - filename containing the RV corrections.
        Columns option a): 0) observation name, rest of columns: corrections with header, to be loaded with pandas dataframe. Header names have to be: 'berv', 'sa', 'drift', 'otherdrift' and optionally the errors. Not all columns are necessary.
        Columns option b): 0) observation name, 1) rv shift (which is considered as 'other drift') 2) rv shift error (optional).
        The columns not present will be 0.


    servalin : str or pd.DataFrame
        Path to directory containing SERVAL data or pandas dataframe with the necessary data (berv, drift and sa) already loaded.
    obj : str
        Tag of the SERVAL outputs, e.g. `obj.rvc.dat`. If None (default), try to get it from the files in `servalin` directly.
    """

    # Get rv corrections from FITS header
    if source == 'header':
        shift, shifterr, datashift = header_rvcorrection_lisobs(lisobs, inst, notfound=notfound, ext=ext)

    # Get rv corrections SERVAL outputs
    elif source == 'serval':
        # This function should also work for HARPS
        print('servalin', servalin)
        shift, shifterr, datashift = carmenesutils.serval_rvcorrection_lisobs(servalin, obj=obj, lisfilobs=lisobs, servalnames=False, use_berv=True, use_drift=True, use_sa=True, join=join)
        # Change index
        datashift['filobs'] = lisobs
        datashift.set_index('filobs', inplace=True)

    # Get rv corrections SERVAL outputs, and if not, from FITS header
    elif source == 'serval_header':
        shift, shifterr, datashift = carmenesutils.serval_caracal_rvcorrection_lisobs(servalin, obj=obj, use_caracal=True, lisfilobs=lisobs, use_berv=True, use_drift=True, use_sa=True, notfound=notfound, ext=ext, verb=True)
        # Change index
        datashift['filobs'] = lisobs
        datashift.set_index('filobs', inplace=True)

    # RV corrections = 0
    elif source == 'none':
        cols = ['berv', 'berverr', 'drift', 'drifterr', 'sa', 'saerr', 'shift', 'shifterr']
        a = np.zeros((len(lisobs), len(cols)))
        datashift = pd.DataFrame(a, columns=cols, index=lisobs)
        shift, shifterr = datashift['shift'], datashift['shifterr']

    # Get rv corrections from file
    elif os.path.exists(source):
        sys.exit('Not implemented yet!')

    else:
        sys.exit('Source {} not correct'.format(source))

    # Add missing columns as nan
    cols = ['berv', 'berverr', 'drift', 'drifterr', 'sa', 'saerr', 'otherdrift', 'otherdrifterr']
    for c in cols:
        if c not in datashift.columns:
            datashift[c] = np.ones_like(shift, dtype=float) * np.nan

    return shift, shifterr, datashift

# -----------------------------------------------------------------------------


# CCF
# ------------------

def fitsccf_rvgrid_read(filin, inst, ext=None, ext1=None):
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        rvgrid, _, _, _, _, _, _, _, _, _ = ccflib.infits_ccfall(filin)
    elif inst == 'HARPS' or inst == 'HARPN':
        if ext is None: ext = 0
        rvgrid = harpsutils.drs_ccfrvgrid_read(filin, ext=ext)
    elif inst == 'ESPRESSO':
        if ext is None: ext = 1
        if ext1 is None: ext1 = 0
        rvgrid = espressoutils.drs_ccfrvgrid_read(filin, ext=ext, ext1=ext1)
    elif inst == 'EXPRES':
        sys.exit()
    return rvgrid


def fitsccf_read_ccf(filin, inst):
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        sys.exit()
    elif inst == 'HARPS' or inst == 'HARPN':
        header, lisccf = harpsutils.drs_ccf_read(filin)
    elif inst == 'ESPRESSO':
        header, lisccf = espressoutils.drs_ccf_read(filin)
    elif inst == 'EXPRES':
        sys.exit()
    return header, lisccf


def fitsccf_fluxes_read(filin, inst):
    if inst == 'CARM_VIS' or inst == 'CARM_NIR':
        _, lisccf, _, _, _, _, _, _, _, _ = ccflib.infits_ccfall(filin)
        lisccferr = np.zeros_like(lisccf) * np.nan
        lisccfq = np.zeros_like(lisccf) * np.nan
    elif inst == 'HARPS' or inst == 'HARPN':
        _, lisccf = harpsutils.drs_ccf_read(filin)
        lisccferr = np.zeros_like(lisccf) * np.nan
        lisccfq = np.zeros_like(lisccf) * np.nan
    elif inst == 'ESPRESSO':
        lisccf, lisccferr, lisccfq = espressoutils.drs_ccffluxes_read(filin)
    elif inst == 'EXPRES':
        sys.exit()
    return lisccf, lisccferr, lisccfq








