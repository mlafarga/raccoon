#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import glob
import os
import sys

import h5py

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

# # FITS
# # ----

# # Spectra and header

# import h5py
# filtest = '/Users/marina/work/data/maroonx/wasp156_stefansson/spectra_20211104/20211104T045821Z_SOOOE_b_0400.hdf'
# # t = Table.read(filtest, path='data')
# data = h5py.File(filtest)

# # data.close()

# """
# # Print level 0 keys
# >>> for x in list(data):
# >>>     print(x, list(data[x]))
# box_extraction ['fiber_1', 'fiber_2', 'fiber_3', 'fiber_4', 'fiber_5']
# etalon_peak_parameters ['peaks', 'polynomials']
# extracted_stripes ['fiber_1', 'fiber_2', 'fiber_3', 'fiber_4', 'fiber_5']
# extraction_parameters ['fiber_1', 'fiber_2', 'fiber_3', 'fiber_4', 'fiber_5']
# header []
# optimal_extraction ['fiber_2', 'fiber_3', 'fiber_4', 'fiber_6']
# optimal_var ['fiber_2', 'fiber_3', 'fiber_4', 'fiber_6']
# stripe_indices ['fiber', 'order']
# wavelengths ['fiber_2', 'fiber_3', 'fiber_4', 'fiber_5', 'fiber_6']
# wavelengths_simultaneous ['fiber_2', 'fiber_3', 'fiber_4', 'fiber_5', 'fiber_6']
# wavelengths_static ['fiber_1', 'fiber_2', 'fiber_3', 'fiber_4', 'fiber_5']

# # Wavelength -> Fiber -> Orders
# >>> for x in list(data['wavelengths']):
# >>>     print(x, list(data['wavelengths'][x]))
# fiber_2 ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '91', '92', '93', '94', '95', '96', '97', '98', '99']
# fiber_3 ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '91', '92', '93', '94', '95', '96', '97', '98', '99']
# fiber_4 ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '91', '92', '93', '94', '95', '96', '97', '98', '99']
# fiber_5 ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '91', '92', '93', '94', '95', '96', '97', '98', '99']
# fiber_6 ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '91', '92', '93', '94', '95', '96', '97', '98', '99']

# # Wavelength values
# >>> data['wavelengths']['fiber_2']['100'][:].shape
# (3954,)
# >>>  data['wavelengths']['fiber_2']['100'][:]
# array([608.18201731, 608.1848439 , 608.1876705 , ..., 617.18155121,
#        617.18335305, 617.18515489])
# """

# lisfiber = ['fiber_2', 'fiber_3', 'fiber_4']
# ordskey = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '91', '92', '93', '94', '95', '96', '97', '98', '99']
# nord = len(ordskey)
# ords = np.arange(0, nord, 1)

# # Read header
# header = dict(data['header'].attrs.items())

# """
#  'FIBER1': 'Sky',
#  'FIBER2': '',
#  'FIBER3': '',
#  'FIBER4': '',
#  'FIBER5': 'Etalon',
# """

# # Read wavelengths
# lisw = [np.vstack([data['wavelengths'][fib][o] for o in ordskey])*10. for fib in lisfiber]  # nm -> A

# # Read fluxes
# lisf = [np.vstack([data['optimal_extraction'][fib][o] for o in ordskey]) for fib in lisfiber]  # nm -> A

# # Read flux uncertainties: optimal_var

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(3, 1, sharex=True)
# for fib in range(len(lisfiber)):
#     for o in ords:
#         ax[fib].plot(lisw[fib][o], lisf[fib][o])
# plt.show(), plt.close()




def drs_bjd_lisobs(lisobs, notfound=np.nan, name='bjd', index=None):
    """

    Returns
    -------
    data : pandas dataframe
    """
    kw = 'JD_UTC_MIDPOINT'
    # float(header['JD_UTC_MIDPOINT'])

    data = {}
    for obs in lisobs:
        with h5py.File(obs, 'r') as filh5:
            header = dict(filh5['header'].attrs.items())
            # print(header['JD_UTC_MIDPOINT'])
            data[obs] = header[kw]
        # filh5 = h5py.File(obs, 'r')
        # header = dict(filh5['header'].attrs.items())
        # filh5.close()

    # Get keywords values
    lisdata = []
    for obs in lisobs:
        dataobs = {}
        with h5py.File(obs, 'r') as filh5:
            header = dict(filh5['header'].attrs.items())
        try: dataobs[kw] = float(header[kw])
        except: dataobs[kw] = notfound
        lisdata.append(dataobs)

    # Dataframe index
    if index is None:
        if isinstance(lisobs[0], str): index = lisobs
        else: index = np.arange(0, len(lisobs), 1)
    # Convert to dataframe
    data = pd.DataFrame(lisdata, index=index)

    # Change dataframe columns
    if name is not None: names = {kw: name}
    else: names = None
    if isinstance(names, dict):
        if list(data.columns) == list(names.keys()):
            data.rename(columns=names, inplace=True)
    
    if name is not None: data[name] = data[name] # + 2400000.
    else: data[kw] = data[kw] # + 2400000.
    return data






