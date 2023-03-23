#!/usr/bin/env python
"""
Compute CCF of a list of observed spectra with a weighted binary mask.
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import textwrap

import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from tqdm import tqdm

from raccoon import ccf as ccflib
from raccoon import carmenesutils
from raccoon import expresutils
from raccoon import peakutils
from raccoon import plotutils
from raccoon import pyutils
from raccoon import spectrographutils
from raccoon import spectrumutils
from raccoon import telluricutils

# Plots
mpl.rcdefaults()
plotutils.mpl_custom_basic()
plotutils.mpl_size_same(font_size=18)

# Constants
C_MS = 2.99792458*1.e8  # Light speed [m/s]
C_KMS = 2.99792458*1.e5  # Light speed [km/s]

###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
        `ccf_compute.py`

        Compute CCF of a list of observed spectra with a weighted binary mask.

        Arguments
        ---------
        '''),
        epilog=textwrap.dedent('''
            '''),
        formatter_class=pyutils.CustomFormatter)

    # Spectra
    parser.add_argument('fil_or_list_spec', help='File with the names of the reduced FITS spectra or directly the file names (names must include the absolute path to the files). The file with the list cannot end in `.fits`.', nargs='+', type=str)
    parser.add_argument('inst', choices=['HARPS', 'HARPN', 'CARM_VIS', 'CARM_NIR', 'EXPRES', 'ESPRESSO', 'ESPRESSO4x2', 'HARPNSOLAR'], help='Instrument.')

    parser.add_argument('--filobs2blaze', help='List of blaze file corresponding to each observation. Format: Column 1) filspec (e2ds), Column 2) filblaze. Full paths. For HARPS/N data. Needed if do not want to use the default from the header. If None, get file names from each observation header.', type=str, default=None)
    # parser.add_argument('--dirblaze', help='Directory containing blaze files. For HARPS/N data.', type=str, default=None)

    parser.add_argument('--expresw', choices=['wavelength', 'bary_wavelength', 'excalibur', 'bary_excalibur'], help='EXPRES wavelength.', default='bary_excalibur')

    # Mask
    parser.add_argument('filmask', help='Path to custom mask file (file with extension `.mas`), or mask ID to use one of the default masks, or (CARM GTO) path to "mask selection file" (file with any other extension that specifies the masks available to choose from), or another type of template (see `--tpltype` option). Mask file format: Columns: 0) w (wavelengths), 1) f (weights), separated by whitespaces. Mask-selection file format: Columns: 0) object used to make the mask `objmask`, 1) spectral type of `objmask`, 2) `vsini` of `objmask`, 3) path to mask file (`.mas` extension). There can only be one mask file for each combination of spt-vsini. TODO: Only spt is used to select the mask (not the vsini).', type=str)
    parser.add_argument('--maskformatharp', help='If mask format is w1, w2, f and wavelengths are in air -> it is transformed into w, f and vacuum.', action='store_true')
    parser.add_argument('--maskair', help='If mask wavelengths in air, tranform to vacuum. Not needed if `maskformatharp` is True.', action='store_true')
    parser.add_argument('--objmask', help='Overwrites values from `filmask`.', type=str, default=None)
    parser.add_argument('--sptmask', help='Overwrites values from `filmask`.', type=str, default=None)
    parser.add_argument('--vsinimask', help='Overwrites values from `filmask`.', type=float, default=None)
    # parser.add_argument('--filmaskabserr')

    # Template
    parser.add_argument('--tpltype', choices=['mask', 'serval', 'phoenix', '1dtxt', 'espressos1dcoadd'], help='', type=str, default='mask')  # 'custommatrix', 'customstepctn'

    parser.add_argument('--tplrv', help='RV shift of the template [km/s] (default is 0)', default=0.0, type=float)

    # MCMC
    parser.add_argument('--mcmc', help='Compute MCMC in logL to get RV. Only if tpltype is not `mask`.')

    # Target info
    parser.add_argument('--obj', help='CARMENES ID.', type=str)
    parser.add_argument('--targ', help='Target name (SIMBAD).', type=str)
    parser.add_argument('--spt', help='Target spectral type. Choices: A) Spectral type with the format `M3.5` (letter and number with 1 decimal), B) `carmencita` to look for the spectral type in the database. Used to select a mask if none is specified. If input not valid, use the default value.', type=str, default='M3.5')
    parser.add_argument('--vsini', help='Target projected rotational velocity [km/s]. Choices: A) float in km/s, e.g. `2.5`, B) `carmencita` to look for the vsini in the database. Used to estimate the CCF RV range. If input not valid or None (default), compute the test CCF and get its width.', type=str, default=None)
    parser.add_argument('--rvabs', help='Absolute RV of the star [km/s]. Used to estimate the centre of the CCF RV array and remove tellurics. If None (default), compute the test CCF and get the RVabs from its minimum. Choices: A) float, B) `carmencita`', type=str, default=None)

    parser.add_argument('--bervmax', help='Maximum BERV to consider when removing mask lines not always visible. Options: A) `obsall` (default): get maximum absolute BERV of all observations available, B) float [m/s]', type=str, default='obsall')

    # CCF computation
    parser.add_argument('--ords_use', nargs='+', help='Sectral orders to consider for the CCF (the orders not listed here will not be used). The orders are counted from 0 (bluest order) to N-1 (reddest order), where N is the number of orders in the template file - so these are not the real orders. Orders in instruments: CARM_VIS: 0-61, CARM_NIR:0-27 (use half order), HARPS: 0-71, HARPN: 0-71. If None (default), all orders are used.', default=None)
    parser.add_argument('--pmin', help='Minimum pixel of each order to use. If None (default) all the pixels are used. Pixels: CARMENES 0-4095, HARP 0-4095, EXPRES 0-7919.', type=int, default=None)
    parser.add_argument('--pmax', help='Maximum pixel of each order to use. If None (default) all the pixels are used.', type=int, default=None)
    parser.add_argument('--wrange', nargs='+', help='Wavelength range to use (2 values), e.g. `--wrange 6000 6500`, [A]. Overwrites ORDS_USE. If None (default), use range defined by orders in ORDS_USE.', type=float, default=None)
    parser.add_argument('--nlinmin', help="Minimum number of usable mask lines per order. Orders with less lines won't be used to compute the CCF.", type=int, default=0)

    # Observations
    parser.add_argument('--obsservalrvc', help='Compute CCFs only of observations in rvc serval files. Must provide SERVAL data.', action='store_true')

    # Extra data
    parser.add_argument('--dirserval', help='Directory containing SERVAL outputs for the observations to analyse. Used to get RV corrections and precise BJD.', default=None)

    # Precise BJD
    parser.add_argument('--bjd', help='BJD source. If `serval`, must provide `dirserval`', choices=['header', 'serval'], default='header')

    # RV corrections
    parser.add_argument('--rvshift', help='Correct spectra RV for BERV, secular acceleration, instrumental drift or other drifts. Options: A) `header`: get corrections (BERV and drift) from FITS header. Secular acceleration and other drift is 0. If nan, use 0. B) `serval`: get corrections (BERV, drift and sa) from SERVAL outputs of the input observations. Must provide SERVAL output directory with `dirserval` (must have run SERVAL previously). C) `serval_header`: get corrections from SERVAL outputs, and if not or nan, get them from header. D) `pathtofile`: File containing the corrections for each obs. Columns: 0) observation name, rest of columns: BERV, drift, sa, other... with header indicating which column is each correction. Can only have a single column with all the corrections already put together. E) `none` (default) then no correction is applied. All in [km/s]', type=str, default=None)

    # Flux correction
    parser.add_argument('--rmvspike', help='Remove flux spikes and interpolate', action='store_true')
    parser.add_argument('--rmvspike_sigma_lower', help='Remove flux spikes: sigma lower', type=float, default=6)
    parser.add_argument('--rmvspike_sigma_upper', help='Remove flux spikes: sigma upper', type=float, default=4)
    parser.add_argument('--rmvspike_maxiters', help='Remove flux spikes: number of iterations', type=int, default=2)
    parser.add_argument('--fcorrorders', help='Correct order flux so that all observations have the same SED', choices=['obshighsnr'], default=None)

    # Telluric mask
    parser.add_argument('--filtell', help='File containing a telluric mask, or `default` to use the default file. If None, no tellurics are removed', type=str, default=None)
    parser.add_argument('--tellbroadendv', help='Velocity by which to broaden the telluric lines to be removed. Options: A) `obsall` (default): get maximum absolute BERV of all observations available, B) float [m/s].', type=str, default='obsall')  # nargs='+',

    # Extra data for CARMENES GTO
    parser.add_argument('--dircarmencita', help='Absolute path.', type=str, default=None)
    parser.add_argument('--carmencitaversion', help='', default=None)

    # CCF test
    parser.add_argument('--ccftestrvcen', help='Central velocity of the RV array of the test CCF [km/s].', type=float, default=0.)
    parser.add_argument('--ccftestrvrng', help='Half of the velocity range of the RV array of the test CCF [km/s].', type=float, default=200.)
    parser.add_argument('--ccftestrvstp', help='Step of the RV array of the test CCF [km/s].', type=float, default=1.)
    parser.add_argument('--ccftesto', help='Order to use to compute the CCF test. If None (default) a specific order depending on the instrument is used: CARM_VIS 36, CARM_NIR X, HARPS and HARPN X.', type=int, default=None)
    parser.add_argument('--ccftestdmin', help='', type=float, default=2.)

    # CCF
    parser.add_argument('--rvcen', help='Central velocity of the RV array of the definitive CCF [km/s] (i.e. absolute RV of the target). If None (default), a test CCF is computed over a broad RV range to find the minimum.', type=float, default=None)
    parser.add_argument('--rvrng', help='Half of the velocity range of the RV array of the definitive CCF [km/s]. If None (default), a test CCF is computed and the range is taken from the test CCF width.', type=float, default=None)
    parser.add_argument('--rvstp', help='Step of the RV array of the definitive CCF [km/s]. This should be smaller than the "real" step RVSTPREAL in order to properly compute the bisector.', type=float, default=0.25)
    parser.add_argument('--rvstpreal', help='Step of the RV array according to the instrument resolution. Needed in order to correctly compute the parameter errors. If None, the default values are used. Better not change this.', type=float, default=None)

    # CCF fit
    parser.add_argument('--fitfunc', help='', type=str, default='gaussian')
    parser.add_argument('--fitrng', help='Range of CCF used to fit the function `fitfunc`. Options: A) float indicating half of the fit range from the CCF minimum [km/s] (e.g. [rvmin - rng, rvmin + rng], B) `maxabs` fit the region between the 2 absolute maxima (default), C) `maxcl` fit the region between the 2 maxima closest to the CCF, D) `all` fit all the CCF range.', type=str, default='maxabs')
    parser.add_argument('--fitrngeach', help='By default (i.e. if this option is not present), if `fitrng` is `maxabs` or `maxcl`, define fit range for all observation using the first one. If `fitrngeach` is present, the maxima are selected for each observation, which may slightly change the fit.', action='store_true')

    # Bisector
    parser.add_argument('--bisectorn', help='', type=int, default=100)
    parser.add_argument('--bisbotmin', help='BIS (bisector inverse slope) bottom region minimum [percent]', type=float, default=10.)
    parser.add_argument('--bisbotmax', help='BIS bottom region maximum [percent]', type=float, default=40.)
    parser.add_argument('--bistopmin', help='BIS top region minimum [percent]', type=float, default=60.)
    parser.add_argument('--bistopmax', help='BIS top region maximum [percent]', type=float, default=90.)

    # logL
    parser.add_argument('--cc2logL', help='Compute logLikelihood (logL) with the CC-to-logL mapping (false if not present).', action='store_true')
    parser.add_argument('--logLmapping', help='Type of CC-to-logL mapping', choices=['zucker2003', 'brogiline2019'])

    # Output
    parser.add_argument('--dirout', help='Output directory.', default='./ccf_output/', type=str)

    parser.add_argument('--output', help='', default=None, type=str, choices=['gto'])

    parser.add_argument('--plot_sv', help='Make and save plots.', action='store_true')
    parser.add_argument('--plot_sh', help='Show all plots.', action='store_true')
    parser.add_argument('--plot_spec', action='store_true')
    # parser.add_argument('--plot_ccfproc', help='CCF process', action='store_true')
    parser.add_argument('--plottest_sh', help='Show test plots to check progress.', action='store_true')
    parser.add_argument('--plottest_sv', help='Save test plots to check progress.', action='store_true')
    parser.add_argument('--plot_ext', nargs='+', help='Extensions of the plots to be saved (e.g. `--plot_ext pdf png`)', default=['pdf'])

    parser.add_argument('--verbose', help='', action='store_true')

    parser.add_argument('--testnobs', help='Testing. Number of observations (>0).', type=int, default=0)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # Verbosity
    verboseprint = print if args.verbose else lambda *a, **k: None

    verboseprint('\n')
    verboseprint('#'*40)
    verboseprint('\nCompute CCF\n')
    verboseprint('#'*40)
    verboseprint('\n')

    # Expand directories and files
    if isinstance(args.fil_or_list_spec, str): args.fil_or_list_spec = os.path.expanduser(args.fil_or_list_spec)
    if isinstance(args.filobs2blaze, str): args.filobs2blaze = os.path.expanduser(args.filobs2blaze)
    if isinstance(args.filmask, str): args.filmask = os.path.expanduser(args.filmask)
    if isinstance(args.dirout, str): args.dirout = os.path.expanduser(args.dirout)
    if isinstance(args.filtell, str): args.filtell = os.path.expanduser(args.filtell)
    if isinstance(args.dirserval, str): args.dirserval = os.path.expanduser(args.dirserval)
    if isinstance(args.dircarmencita, str): args.dircarmencita = os.path.expanduser(args.dircarmencita)

    # Outputs
    if not os.path.exists(args.dirout): os.makedirs(args.dirout)

    # Orders
    carmnirordssplit = True
    nord = spectrographutils.inst_nord(args.inst, carmnirsplit=carmnirordssplit, notfound=None, verb=True)
    ords = np.arange(0, nord, 1)

    # Make sure ords_use is a list-type of ints
    if args.ords_use is not None:
        if not isinstance(args.ords_use, (list, tuple, np.ndarray)): args.ords_use = [args.ords_use]
        if not isinstance(args.ords_use[0], int): args.ords_use = [int(o) for o in args.ords_use]
    else:
        args.ords_use = ords
    args.ords_use = np.sort(args.ords_use)

    # Pixels per order
    if 'CARM' in args.inst or 'HARP' in args.inst:
        # pixmin = 0
        # pixmax = 4096
        npix = 4096
    elif args.inst == 'EXPRES':
        npix = 7920
    elif args.inst == 'ESPRESSO':
        npix = 9111
    elif args.inst == 'ESPRESSO4x2':
        npix = 4545
    pix = np.arange(0, npix, 1)

    # Check extreme pixels to remove inside limits
    if args.pmin is not None:
        if args.pmin < pix[0] or args.pmin > pix[-1]:
            verboseprint('Minimum pixel per order not correct {}. Setting it to {}'.format(args.pmin, pix[0]))
            args.pmin = pix[0]
    else:
        args.pmin = pix[0]

    if args.pmax is not None:
        if args.pmax < pix[0] or args.pmax > pix[-1]:
            verboseprint('Maximum pixel per order not correct {}. Setting it to {}'.format(args.pmax, pix[-1]))
            args.pmax = pix[-1]
    else:
        args.pmax = pix[-1]

    # To plot or not to plot
    doplot = args.plot_sh or args.plot_sv
    doplottest = args.plottest_sh or args.plottest_sv

    # Make sure figure extensions is a list
    if not isinstance(args.plot_ext, list): args.plot_ext = [args.plot_ext]

    if __name__ == "__main__":
        pyutils.save_command_current_hist_args(args.dirout, sys.argv, args)

    ###########################################################################

    # Get reduced spectra
    # -------------------

    # Check if input is file with list or directly the filenames
    # - more than 1 FITS filename in input
    if len(args.fil_or_list_spec) > 1:
        lisfilobs = args.fil_or_list_spec
    # - single FITS filename in input
    elif os.path.splitext(args.fil_or_list_spec[0])[1] == '.fits':
        lisfilobs = args.fil_or_list_spec
    # - file with list in input
    else:
        # Read names of the files
        args.fil_or_list_spec = os.path.expanduser(args.fil_or_list_spec[0])
        lisfilobs = np.loadtxt(args.fil_or_list_spec, dtype='str', usecols=[0])

    # Expand user
    lisfilobs = np.sort([os.path.expanduser(i) for i in lisfilobs])

    if args.testnobs > 0:
        lisfilobs = lisfilobs[:args.testnobs]

    if args.obsservalrvc and args.dirserval:
        # Read SERVAL data
        dataserval = carmenesutils.serval_get(args.dirserval, obj=args.obj, lisdataid=['rvc', 'info'], inst=args.inst)
        # Get observations with non-nan rvc
        mask = np.isfinite(dataserval['servalrvc'])
        dataserval = dataserval[mask]

        # Clean observations
        lisfilobs_new = []
        for i, filobs in enumerate(lisfilobs):
            obs = os.path.basename(filobs)
            if obs in dataserval['obs'].values:
                lisfilobs_new.append(filobs)
        lisfilobs = lisfilobs_new

    # Number of observations
    nobs = len(lisfilobs)
    if nobs == 0:
        sys.exit('No observations found! Exit.')

    # Time ids
    # lisfiltimeid = [i.replace('_A', '') for i in lisfilobs]
    listimeid = [os.path.basename(i).replace('_A.fits', '.fits') for i in lisfilobs]
    listimeid = pd.DataFrame({'timeid': listimeid}, index=lisfilobs)

    # -----------------------------------------------------------------------------

    # Get spectrum data
    # -----------------

    # Get blaze from list (if indicated)
    if args.filobs2blaze is not None:
        lisobs2blaze = np.loadtxt(args.filobs2blaze, dtype=str, usecols=[0, 1], unpack=True)
        # Expand user
        d_obs = [os.path.expanduser(i) for i in lisobs2blaze[0]]
        d_blaze = [os.path.expanduser(i) for i in lisobs2blaze[1]]
        # Convert to dict
        lisobs2blaze = dict(zip(d_obs, d_blaze))
        # Check if files exist
        for k, v in lisobs2blaze.items():
            if not os.path.exists(v):
                verboseprint(' Blaze file {} does not exist!'.format(v))

    # if args.dirblaze is not None:
    #     if isinstance(args.dirblaze, str): args.dirblaze = os.path.expanduser(args.dirblaze)

    # BJD
    if args.bjd == 'header':
        lisbjd = spectrographutils.header_bjd_lisobs(lisfilobs, args.inst, name='bjd', notfound=np.nan, ext=2 if args.inst == 'EXPRES' else 0)
    elif args.bjd == 'serval':
        # Read SERVAL data
        dataserval = carmenesutils.serval_get(args.dirserval, obj=args.obj, lisdataid=['rvc', 'info'], inst=args.inst)
        # Make sure index is timeid, and not bjd
        if dataserval.index.name == 'bjd':
            dataserval['bjd'] = dataserval.index
            dataserval.set_index('timeid', inplace=True)
        # Get BJD of observations
        listimeid_merge = listimeid.copy()
        listimeid_merge['filobs'] = listimeid_merge.index
        listimeid_merge.set_index('timeid', inplace=True, drop=False)
        lisbjd = pd.concat([dataserval.loc[listimeid_merge['timeid'].values, 'bjd'], listimeid_merge], axis=1)[['bjd', 'filobs']]
        lisbjd.set_index('filobs', inplace=True)
    # elif args.bjd == 'serval_header':
    #     pass

    # RV corrections
    if args.rvshift is not None:
        _, _, lisshift = spectrographutils.serval_header_rvcorrection_lisobs(lisfilobs, args.inst, source=args.rvshift, servalin=args.dirserval, obj=args.obj, notfound=np.nan, ext=0)
    else:
        # lisshift = np.zeros_like(lisfilobs, dtype=float)
        shiftcols = ['berv', 'drift', 'berverr', 'drifterr', 'shift', 'shifterr', 'obs', 'sa', 'saerr', 'otherdrift', 'otherdrifterr']
        lisshift = pd.DataFrame(0, columns=shiftcols, index=lisfilobs)

    # Make sure index is lisfilobs
    lisshift['filobs'] = lisfilobs
    lisshift.set_index('filobs', inplace=True)

    # Readout noise RON
    lisron = spectrographutils.header_ron_lisobs(lisfilobs, args.inst, name='ron')

    # SNR per order
    lissnr = spectrographutils.header_snr_lisobs(lisfilobs, args.inst, name='snro', ords=ords)

    # Exposure time
    lisexptime = spectrographutils.header_exptime_lisobs(lisfilobs, args.inst, name='exptime')

    # Airmass
    lisairmass = spectrographutils.header_airmass_lisobs(lisfilobs, args.inst, name='airmass')

    # BERVmax (Maximum abs(BERV) of all the observations)
    if args.bervmax == 'obsall':
        args.bervmax = np.nanmax(np.abs(lisshift['berv']))
    elif pyutils.isfloatnum(args.bervmax):
        args.bervmax = float(args.bervmax)
    else:
        sys.exit('{} BERVmax not valid'.format(args.bervmax))
    verboseprint('  Maximum BERV +-{} m/s'.format(args.bervmax))

    # ---------------------------

    # Merge data into single dataframe
    dataobs = pd.concat([lisbjd, lisshift, lisron, lissnr, lisexptime, lisairmass], axis=1)

    # ---------------------------

    # Reference order
    oref = spectrographutils.inst_oref(args.inst)

    # Reference observation: observation with highest SNR in the reference order
    filobsref = dataobs['snro{:d}'.format(oref)].idxmax()

    # -----------------------------------------------------------------------------

    # Get object characteristics
    # --------------------------

    # Get data from Carmencita

    # Load Carmencita
    if args.rvabs == 'carmencita' or args.vsini == 'carmencita' or args.spt == 'carmencita':
        if not os.path.exists(args.dircarmencita): sys.exit('Carmencita directory {} does not exist'.format(args.dircarmencita))
        datacarmencita = carmenesutils.carmencita_get(args.dircarmencita, version=args.carmencitaversion, verbose=args.verbose)

    # - Absolute RV
    if args.rvabs is not None:
        if args.rvabs == 'carmencita':
            args.rvabs = datacarmencita['Vr_kms-1'].loc[args.obj]  # [km/s]
            if not np.isfinite(args.rvabs):
                args.rvabs = None
        elif pyutils.isfloatnum(args.rvabs):
            args.rvabs = float(args.rvabs)
        else:
            sys.exit('{} not valid'.format(args.rvabs))

    # - vsini
    if args.vsini is not None:
        if args.vsini == 'carmencita':
            args.vsini = datacarmencita['vsini_kms-1'].loc[args.obj]  # [km/s]
            if not np.isfinite(args.vsini):
                args.vsini = None
        elif pyutils.isfloatnum(args.vsini):
            args.vsini = float(args.vsini)
        else:
            sys.exit('{} not valid'.format(args.vsini))

    # - SpT
    if args.spt is not None:
        if args.spt == 'carmencita':
            args.spt = datacarmencita['SpT'].loc[args.obj][:4]
            # if not np.isfinite(args.spt):
            #     args.spt = None
        # elif pyutils.isfloatnum(args.spt):
        #     args.spt = float(args.spt)
        elif len(args.spt) != 4:
            sys.exit('{} not valid. Format must be like `M3.5`.'.format(args.spt))

    verboseprint('Target: {}'.format(args.obj))
    verboseprint('  {} observations'.format(nobs))
    verboseprint('  SpT: {}'.format(args.spt))
    verboseprint('  vsini: {} km/s'.format(args.vsini))
    verboseprint('  RVabs: {} km/s'.format(args.rvabs))
    verboseprint('  Reference order: {}'.format(oref))
    verboseprint('  Reference observation: {}'.format(os.path.basename(filobsref)))
    verboseprint('  Pixel limits: {} -- {}'.format(args.pmin, args.pmax))
    verboseprint('  Orders use: {}'.format(args.ords_use))

    # If trust the values given

    if args.rvabs is not None:
        args.rvcen = args.rvabs

    if args.vsini is not None:
        # Compute rvrng from vsini
        # TODO
        # ...
        pass

    # -----------------------------------------------------------------------------

    # Get template
    # ------------

    filmask = args.filmask
    wm, fm, sfm, nordm, ordsm, headerm = spectrographutils.read_spec_model(filmask, args.tpltype)  # `m` stands for mask or model

    # Flag bad orders in model
    # ------------------------

    # If order empty -> Do not use, remove from args.ords_use
    omempty = [o for o in ordsm if len(wm[o]) == 0]
    # maskoempty = ~np.asarray([len(w[o])==0 if o in args.ords_use else False for o in ords])
    # args.ords_use = args.ords_use[maskoempty]

    # If wavelength or flux of an order is all nan -> Do not use, remove from args.ords_use
    omwnan = [o for o in ordsm if ~np.isfinite(wm[o]).all()]
    omfnan = [o for o in ordsm if ~np.isfinite(fm[o]).all()]

    # If wavelength or flux of an order is all 0 -> Do not use, remove from ordsm
    omw0 = [o for o in ordsm if not np.any(wm[o])]
    omf0 = [o for o in ordsm if not np.any(fm[o])]

    # Total orders to remove
    omrmv = np.sort(np.unique(omempty + omwnan + omfnan + omw0 + omf0))
    ordsm_use_new = []
    for o in ordsm:
        if o in omrmv:
            continue
        else:
            ordsm_use_new.append(o)
    ordsm_use = np.asarray(ordsm_use_new, dtype=int)
    nordm_use = len(ordsm_use)

    # Shift model if `tplrv` is != 0
    if args.tplrv != 0.0:
        if nordm > 1:
            wm = np.array([wm[o] * (1 - args.tplrv / C_KMS) if o in ordsm_use else wm[o] for o in range(len(wm))])
            # wm = np.concatenate(wm).reshape(61,)

            # for o in range(len(wm)):
            #     if o in ordsm_use:
            #         wm[o] * (1 - args.tplrv / C_KMS)
            #     else:
            #         wm 
        else:
            wm = wm * (1 - args.tplrv / C_KMS)

    ###########################################################################

    # CCF test: Determine CCF RV center and range
    # -------------------------------------------

    # !!!!!!!!!!!need to provide args.rvcen and args.rvrng manually

    # Determine RV steps

    # - Oversampled (for bisector)
    verboseprint('  RV steps oversampled: {} km/s'.format(args.rvstp))

    # - Real (for errors)
    if args.rvstpreal is None:
        args.rvstpreal = spectrographutils.inst_rvpixmed(args.inst)
        verboseprint('  RV steps real sampling: {} km/s'.format(args.rvstpreal))
    else:
        verboseprint('  -- RV steps for real sampling fixed by user at {} km/s, NOT recommended!'.format(args.rvstpreal))

    if args.rvstp >= args.rvstpreal:
        verboseprint('  -- RV steps oversampled: {} >= RV steps real: {} -> May have problems with bisector!'.format(args.rvstp, args.rvstpreal))
        pass

    # ---------------------------------

    # Final RV arrays
    rv = np.arange(args.rvcen - args.rvrng, args.rvcen + args.rvrng + args.rvstp, args.rvstp)
    rvreal = np.arange(args.rvcen - args.rvrng, args.rvcen + args.rvrng + args.rvstpreal, args.rvstpreal)

    ###########################################################################

    # Get order relative flux correction
    # -----------------------------------

    # Flux correction obtained from reference observation
    # so that the flux of the orders orders of all the observations always
    # has the same ratio

    if args.fcorrorders == 'obshighsnr':
        filblaze = lisobs2blaze[filobsref] if args.filobs2blaze is not None else None
        w, f, sf, c, header, dataextra = spectrographutils.fitsred_read(filobsref, args.inst, carmnirdiv=carmnirordssplit, harpblaze=True, dirblaze=None, filblaze=filblaze, expresw=args.expresw)

        # EXPRES use own mask to remove bad pixels
        if args.inst == 'EXPRES':
            w, f, sf, c = expresutils.apply_expres_masks_spec(w, f, sf, c, dataextra['pixel_mask'], excalibur_mask=dataextra['excalibur_mask'] if args.expresw == 'bary_excalibur' or args.expresw == 'excalibur' else None)

        # Cont
        f = [f[o]/c[o] for o in ords]
        # flux_ratios2 = [np.nanmedian(f[o]) / np.nanmedian(f[oref]) for o in ords]
        # SNR
        if args.inst == 'CARM_VIS' or args.inst == 'CARM_NIR':
            f = [f[o] * dataobs.loc[filobsref]['snro{:d}'.format(o)]**2 / np.nanmedian(f[o]) for o in ords]

        # flux_ratios = [np.nanmedian(f[o]) / np.nanmedian(f[oref]) for o in ords]
        flux_ratios = [np.nanmedian(f[o]) / np.nanmedian(f[oref]) for o in ords]
        # plt.plot(ords, flux_ratios, 'o', ords, flux_ratios2, 'x'), plt.show(), plt.close()

        # Save flux ratios per order: Order, flux ratio (i.e. weight), lambda min order, lambda max order
        if args.output is None:
            arrwmin = [np.nanmin(w[o]) for o in ords]
            arrwmax = [np.nanmax(w[o]) for o in ords]
            filout = os.path.join(args.dirout, '{}.flux_ratios.dat'.format(args.obj))
            np.savetxt(filout, np.vstack((ords, flux_ratios, arrwmin, arrwmax)).T, delimiter=' ', fmt=['%d', '%.8f', '%.8f', '%.8f'])
        elif args.output == 'gto':
            pass

    ###########################################################################

    # Select mask lines free of tellurics
    # -----------------------------------

    # Correct for tellurics
    if args.filtell is not None:

        verboseprint('\nTelluric mask:', args.filtell)

        if args.filtell == 'default':
            args.filtell = ccflib.selecttell_default(args.inst)
            verboseprint('  ', args.filtell)

        # Tellurics
        wt, ft = telluricutils.read_mask(args.filtell)
        wt1, wt2 = telluricutils.mask2wlimits(wt, ft)

        # Get broaden velocity
        if args.tellbroadendv == 'obsall':
            args.tellbroadendv = np.nanmax(np.abs(lisshift['berv']))
        elif pyutils.isfloatnum(args.tellbroadendv):
            args.tellbroadendv = float(args.tellbroadendv)
        else:
            sys.exit('{} tellbroadendv not valid'.format(args.tellbroadendv))
        verboseprint('  Broaden telluric features by +-{} m/s'.format(args.tellbroadendv))

        # Broaden
        wt_broaden = telluricutils.broaden_mask(wt, ft, args.tellbroadendv)
        wt1_broaden, wt2_broaden = telluricutils.broaden_wlimits(wt1, wt2, args.tellbroadendv)

        # Join overlaping lines
        wt1_broaden_join, wt2_broaden_join = telluricutils.join_overlap_wlimits(wt1_broaden, wt2_broaden)
        # Make mask with joined limits
        wt_broaden_join, ft_broaden_join = telluricutils.wlimits2mask(wt1_broaden_join, wt2_broaden_join, dw=0.001)

        # -----------------------

        # Telluric mask -> Function

        # Extend mask (with flux 0) to all the spectrum range, so there are no interpolation problems
        try:
            wspecmin, wspecmax = min(np.concatenate(w)), max(np.concatenate(w))
        except:
            filblaze = lisobs2blaze[filobsref] if args.filobs2blaze is not None else None
            w, f, sf, c, header, dataextra = spectrographutils.fitsred_read(filobsref, args.inst, carmnirdiv=carmnirordssplit, harpblaze=True, dirblaze=None, filblaze=filblaze, expresw=args.expresw)
            # Transform spectra to vacuum (only HARPS/N)
            if args.inst == 'HARPS' or args.inst == 'HARPN':
                w = spectrumutils.wair2vac(w)
            wspecmin, wspecmax = min(np.concatenate(w)), max(np.concatenate(w))

        # wt_broaden_join = np.concatenate(([wspecmin], wt_broaden_join, [wspecmax]))
        if nordm > 1: 
            wm0 = np.nanmin(wm[ordsm_use])
            wm1 = np.nanmax(wm[ordsm_use])
        else:
            wm0 = wm[0]
            wm1 = wm[-1]
        wmin = np.nanmin([wspecmin, wm0, wt_broaden_join[0]]) - 500.
        wmax = np.nanmax([wspecmax, wm1, wt_broaden_join[-1]]) + 500.
        wt_broaden_join = np.concatenate(([wmin], wt_broaden_join, [wmax]))
        ft_broaden_join = np.concatenate(([0], ft_broaden_join, [0]))

        # Original masks: 0 good regions, 1 regions to be masked
        # Invert masks: 0 to bad, 1 to good
        Maskbad_inv, fmaskbad_inv = telluricutils.interp_mask_inverse(wt_broaden_join, ft_broaden_join, kind='linear')

        # -----------------------

        # Remove mask lines in telluric regions. Take into account velocity of the star and mask shift

        # Get mask lines affected by tellurics for each mask shift
        # Shift step should be at least the minimum width of a telluric line, about 2*BERVmax km/s, so that no overlaps are missed.
        # Here use `rvstp`, which is enough for sure

        # Make sure rvstp is small enough
        # If not make it half of the BERV, to be sure that no overlaps are missed
        if rv[1] - rv[0] > 2 * args.bervmax / 1.e3:
            # TODO: fails is args.bermax is 0!!!
            rvnew = np.arange(rv[0], rv[-1] + (rv[1] - rv[0]), args.bervmax / 2.e3)
        else:
            rvnew = rv

        # Binary masks
        if nordm == 1:
            mask = [[]] * len(rvnew)
            for i, maskshift in enumerate(rvnew):
                wm_shift = wm * (1 + maskshift / C_KMS)  # works for 1D and 2D wm

                # Flag lines in bad regions: False in bad region, True otherwise
                mask[i] = np.array(Maskbad_inv(wm_shift), dtype=bool)

            # Join masks for each RV shift
            maskjoin = np.prod(mask, axis=0, dtype=bool)

            # # Plot: Check mask join is OK
            # for i in range(len(mask)):
            #     plt.plot(np.arange(len(mask[0])), mask[i], 'o')
            # plt.plot(np.arange(len(mask[0])), maskjoin, '+', ms=10)
            # plt.show()
            # plt.close()

            # Remove lines in telluric regions
            wm = wm[maskjoin]
            fm = fm[maskjoin]

            nlin = len(wm)

            verboseprint('Remove mask lines affected by tellurics')
            verboseprint('  {} lines in mask after removing tellurics'.format(nlin))

        # Full templates: Cannot really remove tellurics (make them nan or 0) because the template is interpolated and there's issues with that. Plus we also need the mean to having tellurics set as 0 causes issues.
        # Instead of making them 0, keep model as is, and keep track of the bad wavelength with the function Maskbad_inv above. Remove them for each RV shift when actually computing the CCF
        else:
            """
            for o in ordsm_use:
                mask = [[]] * len(rvnew)
                for i, maskshift in enumerate(rvnew):
                    wm_shift = wm[o] * (1 + maskshift / C_KMS)  # works for 1D and 2D wm

                    # Flag lines in bad regions: False in bad region, True otherwise
                    mask[i] = np.array(Maskbad_inv(wm_shift), dtype=bool)

                # Join masks for each RV shift
                maskjoin = np.prod(mask, axis=0, dtype=bool)

                # # Remove pixels in telluric regions -> Issues is wm is an array, changing dimensions of each row gives problems
                # wm[o] = wm[o][maskjoin]
                # fm[o] = fm[o][maskjoin]
                # # Make bad pixels nan to conserve array shape -> Issues when inter/extrapolating with nan
                # wm[o][~maskjoin] = np.nan
                # fm[o][~maskjoin] = np.nan

                # Conserve wm as is, make flux 0 so it doesn't contribute to the CCF
                # ---> Interpolating issues
                fm[o][~maskjoin] = 0.0
                
                # # Set to nan
                # # -> Issues when interpolating
                # fm[o][~maskjoin] = np.nan

                # Keep track of the bad wavelength with the mask function Maskbad_inv

                # nlin = len(wm)
            """

            # Include the RVshifts in the Maskbad_inv function (now only contains the BERV shifts)
            # TODO

        # # Plot telluric regions, mask lines and mask lines removed
        # fig, ax = plt.subplots()
        # ax.vlines(wm_all, 0, fm_all/np.nanmax(fm_all), colors='C2', label='Mask')
        # ax.plot(wm_all[~mask], fm_all[~mask]/np.nanmax(fm_all), 'C1x', mew=2, label='Removed lines')
        # ax.fill_between(wt_broaden_join, ft_broaden_join, color='.5', label='Tellurics', zorder=0)
        # plt.show()
        # plt.close()

    # -----------------------------------------------------------------------------


    # Select mask lines per order usable at any epoch
    # -----------------------------------------------

    verboseprint('\nRemove mask lines at order edges, taking into account BERV and mask shift')


    # Order wavelength limits

    # - General limits

    #   Read from file
    #   TODO

    #   or

    #   Get them from obsref
    # w, f, sf, c, header, dataextra = spectrographutils.fitsred_read(filobsref, args.inst, carmnirdiv=carmnirordssplit, harpblaze=True, dirblaze=None, filblaze=None, expresw=args.expresw)
    filblaze = lisobs2blaze[filobsref] if args.filobs2blaze is not None else None
    # if args.filobs2blaze is not None:
    # filblaze = lisobs2blaze[filobs]
    # else:
    # filblaze = None
    w, _, _, _, _, _ = spectrographutils.fitsred_read(filobsref, args.inst, carmnirdiv=carmnirordssplit, harpblaze=True, dirblaze=None, filblaze=filblaze, expresw=args.expresw)
    # Transform spectra to vacuum (only HARPS/N)
    if args.inst == 'HARPS' or args.inst == 'HARPN':
        w = spectrumutils.wair2vac(w)

    # Remove extreme pixels
    if args.pmax < len(pix):
        pmaxp = args.pmax + 1
    elif args.pmax == len(pix):
        pmaxp = args.pmax

    w = w[:, args.pmin:pmaxp]
    # f = f[:, args.pmin:pmaxp]
    # sf = sf[:, args.pmin:pmaxp]
    # c = c[:, args.pmin:pmaxp]
    # if args.inst == 'EXPRES':
    #     dataextra['pixel_mask'] = dataextra['pixel_mask'][:, args.pmin:pmaxp]
    #     dataextra['excalibur_mask'] = dataextra['excalibur_mask'][:, args.pmin:pmaxp]

    # # EXPRES use own mask to remove bad pixels
    # if args.inst == 'EXPRES':
    #     w = expresutils.apply_expres_masks_array(w, dataextra['pixel_mask'], excalibur_mask=dataextra['excalibur_mask'] if args.expresw == 'bary_excalibur' or args.expresw == 'excalibur' else None)

    # EXPRES change w=0 to w=nan
    if args.inst == 'EXPRES':
        w[w == 0.] = np.nan

    womin = np.nanmin(w, axis=1)
    womax = np.nanmax(w, axis=1)

    # ----

    # Order limits taking into account maximum BERV and maximum mask shift

    womaxshift = womax * (1 - (args.bervmax*1.e-3 + np.nanmax(rv)) / C_KMS)
    wominshift = womin * (1 - (-args.bervmax*1.e-3 + np.nanmin(rv)) / C_KMS)

    # ----

    # Lines per order usable at any epoch

    wmords, fmords = [[]]*nord, [[]]*nord
    if nordm > 1:
        for o in ords:
            mask = (wm[o] >= wominshift[o]) & (wm[o] <= womaxshift[o])
            wmords[o] = wm[o][mask]
            fmords[o] = fm[o][mask]
    else:
        for o in ords:
            mask = (wm >= wominshift[o]) & (wm <= womaxshift[o])
            wmords[o] = wm[mask]
            fmords[o] = fm[mask]

    # ----

    # Remove orders with not enough lines
    ords_use_new = []
    # Number of lines per order
    nlinords = [len(o) for o in wmords]
    ords_keep = np.argwhere(np.array(nlinords) >= args.nlinmin).flatten()
    # args.ords_use = np.array(list(set(args.ords_use).intersection(ords_keep)))
    args.ords_use = np.sort(list(set(args.ords_use).intersection(ords_keep)))

    # ----

    # Mask lines all orders (count overlaps too)
    wmall = np.hstack(wmords)
    fmall = np.hstack(fmords)

    # Mask lines that will be used (count overlaps too) i.e. lines in ords_use
    wmall_use = np.hstack([wmords[o] for o in args.ords_use])
    fmall_use = np.hstack([fmords[o] for o in args.ords_use])

    # Number of lines per order
    nlinords = [len(o) for o in wmords]

    # fig, ax = plt.subplots()
    # for o in args.ords_use:
    #     ax.plot(w[o], f[o])
    #     ax.vlines(wmords[o], 0, fmords[o])
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    # sys.exit()

    # Total number of lines (including duplicates due to order overlap)
    nlin = len(wmall)
    nlin_use = len(wmall_use)

    verboseprint('  {} lines in mask after removing tellurics and order edges'.format(nlin))
    verboseprint('  {} lines in mask that will be used'.format(nlin_use))
    verboseprint('  -> Some lines may be duplicated because of order overlap!')
    verboseprint('  Lines per order:')
    verboseprint('     Ord    Nlin')
    for o in ords:
        verboseprint('    {:4d}   {:5d}'.format(o, nlinords[o]))
    verboseprint('  Orders with less than {} lines will not be used'.format(args.nlinmin))
    verboseprint('  Orders use update: {}'.format(args.ords_use))

    ###########################################################################

    # Compute CCF

    # --- Start observations loop ---
    # ccf, ccferr, ccfreal, ccferrreal = {}, {}, {}, {}
    # ccfpar = {}
    dataccfsumTS = {}
    # dataccfoTS = {}
    dataccfoTS = []
    first, firsto = True, True
    for i, obs in enumerate(tqdm(lisfilobs)):
        filobs = lisfilobs[i]
        obsid = os.path.basename(os.path.splitext(filobs)[0])
        # verboseprint('{}/{} {}'.format(i+1, nobs, filobs))

        # Get blaze file (if indicated)
        if args.filobs2blaze is not None:
            filblaze = lisobs2blaze[filobs]
        else:
            filblaze = None

        # Read spectrum
        w, f, sf, c, header, dataextra = spectrographutils.fitsred_read(filobs, args.inst, carmnirdiv=True, harpblaze=True, dirblaze=None, filblaze=filblaze, expresw=args.expresw)
        # Transform spectra to vacuum (only HARPS/N)
        if args.inst == 'HARPS' or args.inst == 'HARPN':
            w = spectrumutils.wair2vac(w)
        wraw, fraw, sfraw, craw = w.copy(), f.copy(), sf.copy(), c.copy()

        # Plot spectrum
        if args.plot_spec:
            fig, ax = plt.subplots(2, 1)
            # for o in args.ords_use:
            for oo in ords:
                if oo % 2 == 0: c = 'k'
                else: c = '.5'
                ax[0].plot(w[oo], f[oo], linewidth=0.5, marker='.', color=c, alpha=0.8)
                ax[1].plot(w[oo], f[oo]/c[oo], linewidth=0.5, marker='.', color=c, alpha=0.8)
                # ax[1].plot(w[o], f[o]/c[o], linewidth=0.5, marker='.', color=c)
            plt.tight_layout()
            plt.show()
            plt.close()

        # Apply ESPRESSO quality mask (basically seems to remove pixels with0 flux values at the extremes)
        w_new, f_new, sf_new, c_new = [], [], [], []
        if args.inst == 'ESPRESSO' or args.inst == 'ESPRESSO4x2':
            for o in range(len(w)):
                mq = dataextra['mq'][o]
                w_new = w[o][mq]
                f_new = f[o][mq]
                sf_new = sf[o][mq]
                c_new = c[o][mq]
            w_new, f_new, sf_new, c_new = w, f, sf, c

        # Remove extreme pixels
        w = w[:, args.pmin:pmaxp]
        f = f[:, args.pmin:pmaxp]
        sf = sf[:, args.pmin:pmaxp]
        c = c[:, args.pmin:pmaxp]


        # EXPRES use own mask to remove bad pixels
        if args.inst == 'EXPRES':
            w, f, sf, c = expresutils.apply_expres_masks_spec(w, f, sf, c, dataextra['pixel_mask'][:, args.pmin:pmaxp], excalibur_mask=dataextra['excalibur_mask'][:, args.pmin:pmaxp] if args.expresw == 'bary_excalibur' or args.expresw == 'excalibur' else None)

        # Remove pixels with nan or negative wavelengths
        spec_removenan, masknan = spectrumutils.remove_nan_echelle(w, f, sf, c, ords_use=args.ords_use, returntype='original')
        w, f, sf, c = spec_removenan

        # More cleaning...
        # - Pixels with nan or negative wavelengths
        # - Zeros at order edges

        # if i == 19:
        # # if i == 0 or i == 19:
        #     fig, ax = plt.subplots(2, 1)
        #     # for o in args.ords_use:
        #     for oo in ords:
        #         if oo % 2 == 0: c = 'k'
        #         else: c = '.5'
        #         ax[0].plot(w[oo], f[oo], linewidth=0.5, marker='.', color=c)
        #         # ax[1].plot(w[o], f[o]/c[o], linewidth=0.5, marker='.', color=c)
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()

        # Correct flux
        f = [f[o]/c[o] for o in ords]
        sf = [sf[o]/c[o] for o in ords]

        # CARMENES: c = 1
        if args.inst == 'CARM_VIS' or args.inst == 'CARM_NIR' or args.inst == 'EXPRES':
            c = [np.ones(len(w[o])) for o in ords]

        # HARPS correct order slope with a line
        if args.inst == 'HARPS' or args.inst == 'HARPN':
            # TODO: fit only line max
            for o in ords:
                SpecFitPar = np.polyfit(w[o], f[o], 1)
                f[o] = np.array(f[o] / np.poly1d(SpecFitPar)(w[o]) * np.mean(f[o]))
            # fig, ax = plt.subplots(4, 1, figsize=(8, 12))
            # for o in [25, 26, 27]:
            #     ax[0].plot(wraw[o], fraw[o]/craw[o])
            #     ax[1].plot(w[o], c[o])
            #     ax[2].plot(w[o], fraw[o]*c[o])
            #     ax[3].plot(w[o], f[o])
            # plt.tight_layout()
            # plt.show()
            # plt.close()
            # sys.exit()

        # Clip flux spikes
        if args.rmvspike:
            fnew = []
            for o in ords:
                # Mask with spikes (True)
                maskclip = spectrumutils.mask_spike(f[o], sigma_lower=args.rmvspike_sigma_lower, sigma_upper=args.rmvspike_sigma_upper, maxiters=args.rmvspike_maxiters, axis=0)
                # Make points near spikes (True) also maked (True)
                maskclip = spectrumutils.extend_mask_1D(maskclip, pointnear=1)
                # Correct spikes by interpolating
                fclip = spectrumutils.clean_spike_1D(w[o], f[o], maskclip, clean='interpol')
                # plt.plot(w[o], f[o], '-', w[o], fclip, '--'), plt.show()
                fnew.append(fclip)
            f = fnew


        # CARMENES (CARACAL) FOX: Reweight flux so that <counts> = SNR^2
        if args.inst == 'CARM_VIS' or args.inst == 'CARM_NIR' or args.inst == 'EXPRES' or args.inst == 'ESPRESSO' or args.inst == 'ESPRESSO4x2':
            f = [f[o] * dataobs.loc[filobs]['snro{:d}'.format(o)]**2 / np.nanmean(f[o]) for o in ords]
            # f2 = [f[o] * dataobs.loc[filobs]['snro{:d}'.format(o)]**2 / np.median(f[o]) for o in ords]

        # fig, ax = plt.subplots(4, 1, sharex=True)
        # for o in ords:
        #     ax[0].plot(wraw[o], fraw[o])
        #     ax[1].plot(w[o], f[o])
        #     # ax[2].plot(w[o], f2[o])
        # lissnrobs = [dataobs.loc[filobs]['snro{:d}'.format(o)] for o in ords]
        # liswcen = [np.nanmean(w[o]) for o in ords]
        # ax[3].plot(liswcen, lissnrobs, 'o')
        # plt.show()
        # plt.close()

        # Apply order relative flux correction
        if args.fcorrorders == 'obshighsnr':
            fsnroriginal = f.copy()
            flux_ratios_obs = [np.nanmedian(f[o]) / np.nanmedian(f[oref]) for o in ords]
            f = [f[o] * flux_ratios[o] / flux_ratios_obs[o] for o in ords]
            # fcorr = [f[o] * flux_ratios[o] / flux_ratios_obs[o] for o in ords]

            # # Check flux ratios
            # flux_ratios_check = [np.nanmedian(fcorr[o]) / np.nanmedian(fcorr[oref]) for o in ords]
            # for o in ords:
            #     print(flux_ratios[o], flux_ratios_check[o], flux_ratios_obs[o])
            # fig, ax = plt.subplots(4, 1)#, sharex=True)
            # for o in ords:
            #     ax[0].plot(wraw[o], fraw[o])
            #     ax[1].plot(w[o], f[o])
            #     ax[2].plot(w[o], fcorr[o])
            # ax[3].plot(ords, flux_ratios, 'o')
            # ax[3].plot(ords, flux_ratios_check, 'x')
            # plt.tight_layout()
            # plt.show()
            # plt.close()
        else:
            fsnroriginal = None
            #fsnroriginal = [[None]]*len(f)

        # Correct spectrum wavelength: BERV, drift, secular acceleration
        shift = dataobs.loc[filobs]['shift']
        if np.isfinite(shift): shift_use = shift
        else: shift_use = 0.
        shifterr = dataobs.loc[filobs]['shifterr']
        if np.isfinite(shifterr): shifterr_use = shifterr
        else: shifterr_use = 0.
        wcorr = [w[o] * (1. - (shift_use) / C_MS) for o in ords]

        # Readout noise
        ron = dataobs.loc[filobs]['ron']


        #######################################################################

        # Compute CCF with full model and logL (CC-to-logL)
        # -------------------------------------------------

        # Use only orders where the flux is not all zeros
        # ords_use_lines = [o for o in args.ords_use if len(wmords[o]) > 0]
        ords_use_lines = [o for o in args.ords_use if np.any(fmords[o])]

        # Whole template interpolation
        # Interpolation coefficients of the template
        #   If model is 1D: single list of coefficients, if multiple orders, multiple lists of coefficients
        # if nordm > 1:
        #     liscs = [splrep(wm[om], fm[om], s=0) for om in ordsm]
        # else:
        #     cs = splrep(wm, fm, s=0)

        # ipdb.set_trace()
        # for om in ordsm:
        #     print(om)
        #     if om in ords_use_lines:
        #         splrep(wmords[om], fmords[om], s=0)

        # # No yerr
        # liscs = [splrep(wmords[om], fmords[om], s=0) if om in ords_use_lines else np.nan for om in ordsm ]
        liscs = [splrep(wmords[om], fmords[om], k=1) if om in ords_use_lines else np.nan for om in ordsm ]
        # liscs = [splrep(wmords[om], fmords[om], w=1./sfm[o], s=0) if om in ords_use_lines else np.nan for om in ordsm ]  # using weight w results in all nan when interpolating
        # TODO: This can be done outside the observation loop


        # Variables to store data
        cc = np.empty((nord, len(rv)))*np.nan
        logLZ03, sigZ03 = np.empty((nord, len(rv)))*np.nan, np.empty((nord, len(rv)))*np.nan
        rvmaxZ03, rvmaxerrZ03, rvmaxerrlZ03, rvmaxerrrZ03 = np.empty(nord)*np.nan, np.empty(nord)*np.nan, np.empty(nord)*np.nan, np.empty(nord)*np.nan
        logLBL19, sigBL19 = np.empty((nord, len(rv)))*np.nan, np.empty((nord, len(rv)))*np.nan
        rvmaxBL19, rvmaxerrBL19, rvmaxerrlBL19, rvmaxerrrBL19 = np.empty(nord)*np.nan, np.empty(nord)*np.nan, np.empty(nord)*np.nan, np.empty(nord)*np.nan
        
        # --- Start orders loop ---
        for o in ords_use_lines:
        # for o in [35]:

            # Function to make tellurics 0
            mtell_obs = np.array(Maskbad_inv(wcorr[o]), dtype=bool)
            # TODO: this should also include order extremes

            # Number of datapoints
            # TODO: Should be without tellurics
            N_full = len(f[o])
            N = len(f[o][mtell_obs])  # accounting for tellurics
            Id = np.ones(N_full)  # for matrix operations
            
            fVec = f[o].copy()

            # Make tellurics zero
            fVec[~mtell_obs] = 0.0

            # Stdev of the spectrum
            fVec -= (fVec[mtell_obs] @ Id[mtell_obs]) / N  # subtract mean
            sf2 = (fVec[mtell_obs] @ fVec[mtell_obs]) / N  # stdev spectrum

            # Temporal variables to append Doppler-shift values
            cc_i, logLZ03_i, logLBL19_i = [], [], []

            # For each RV shift in the CCF grid
            for irvshift, rvshift in enumerate(rv):

                # Fast CCF: shift observation w minus the RV shift of the CCF
                # --------

                # Doppler shift obs w minus rvshift
                wobs_shift = spectrumutils.dopplershift(wcorr[o], -rvshift * 1.e3, rel=True)
                
                # # First and last orders, check if w observation out of model range, and cut accordingly. If not, will have issues with the logL
                # if io == iords_use[0] or io == iords_use[1]:
                #     pass
                # if io == iords_use[-2] or io == iords_use[-1]:
                #     if wobs_shift[-1] > wtpl[-1]:
                #         mask = wobs_shift > wtpl[-1]
                #         wobs_shift[mask] = np.nan

                # Interpolate model to shifted obs w
                if nordm > 1: cs = liscs[o]
                fm_obsgrid = splev(wobs_shift, cs, der=0, ext=3)  # ext=0 means return extrapolated value, if ext=3, return the boundary value
                # ISSUE TODO: Different lines (pixels) for different shifts. Extremes (+- CCF RV shift (and BERV)) should be set to 0.

                # Make tellurics nan
                # mtell_i = np.array(Maskbad_inv(wobs_shift), dtype=bool)
                # fm_obsgrid[~mtell_i] = 0.0
                # fm_obsgrid[~mtell_obs] = 0.0  # -> Should be mtell_obs because tellurics do not move

                # if np.array_equiv(~mtell_obs, ~mtell_i) is False:
                #     print('----', i, o, irvshift, np.array_equiv(mtell_obs, mtell_i))
                # ipdb.set_trace()

                # Stdev of the model
                gVec = fm_obsgrid.copy()
                gVec[~mtell_obs] = 0.0
                gVec -= (gVec[mtell_obs] @ Id[mtell_obs]) / N  # subtract mean  ----------> CAREFUL WITH TELLURICS
                sg2 = (gVec[mtell_obs] @ gVec[mtell_obs]) / N  # stdev model

                """
                if irvshift == 0 and (o == 25 or o == 35 or o == 43):
                    fig, ax = plt.subplots(2,1, figsize=(16,8), sharex=True)
                    # ax[0].plot(wcorr[o], f[o])
                    ax[0].plot(wcorr[o], fVec, label='sf2 {:.3f}'.format(sf2))
                    ax[1].plot(wmords[o], fmords[o]) 
                    ax[1].plot(wobs_shift, fm_obsgrid) 
                    ax[1].plot(wobs_shift, gVec, label='sg2 {:.3f}'.format(sg2))
                    for a in ax: a.legend()
                    plt.tight_layout()
                    plt.show(), plt.close()
                """


                # Cross-covariance function
                R = (fVec[mtell_obs] @ gVec[mtell_obs]) / N
                # Compute the CCF between the obs f and the interpolated model f
                cc_rv = R / np.sqrt(sf2*sg2)

                # Compute logL Z03
                logLZ03_rv = - N/2. * np.log(1 - cc_rv**2)

                # Compute logL BL19
                logLBL19_rv = - N/2. * np.log(sf2 + sg2 - 2.*R)

                # Save
                cc_i.append(cc_rv)
                logLZ03_i.append(logLZ03_rv)
                logLBL19_i.append(logLBL19_rv)
            # --- End RV shift loop ---

            # Save
            cc[o] = cc_i
            logLZ03[o] = logLZ03_i
            logLBL19[o] = logLBL19_i

            # Compute sigma
            sigZ03[o], pZ03, dlogLZ03 = ccflib.logL2sigma(logLZ03[o], dof=1)
            sigBL19[o], pBL19, dlogLBL19 = ccflib.logL2sigma(logLBL19[o], dof=1)

            # ---------------

            # # Plot CC, logL and sigma
            # ccflib.plot_cc_logLZ03_logLBL19_sig(rv, cc[o], logLZ03[o], logLBL19[o], sigZ03[o], sigBL19[o], title='Order {}'.format(o))

            # ---------------

            # RV from maximum logL
            rvmaxZ03[o] = rv[np.argmax(logLZ03[o])]
            rvmaxBL19[o] = rv[np.argmax(logLBL19[o])]

            # ---------------

            # Compute RV err
            # # --- Mask: sigma can be inf/nan. This gives problems with the interpolation after
            # mask_sigZ03 = np.isfinite(sigZ03[o])
            # # --- Interpolate sigma function left and right hand sides to get dRV of sigma1 - sigma0. Compute average between left and right
            # imin = np.argmin(sigZ03[o][mask_sigZ03])
            # rvsig0 = rv[mask_sigZ03][imin]
            # # ------ Left
            # RVfunc_l = interp1d(sigZ03[o][mask_sigZ03][:imin+1], rv[mask_sigZ03][:imin+1], kind='cubic')
            # rvsig1_l = RVfunc_l(1)
            # rvmaxerrlZ03 = np.abs(rvsig1_l - rvsig0)
            # # ------ Right
            # RVfunc_r = interp1d(sigZ03[o][mask_sigZ03][imin:], rv[mask_sigZ03][imin:], kind='cubic')
            # rvsig1_r = RVfunc_r(1)
            # rvmaxerrrZ03 = np.abs(rvsig1_r - rvsig0)
            # # ------ Average
            # rvmaxerrZ03 = np.nanmean([rvmaxerrlZ03, rvmaxerrrZ03])
            rvmaxerrZ03[o], rvmaxerrlZ03[o], rvmaxerrrZ03[o] = ccflib.rverr_from_sig(rv, sigZ03[o])
            rvmaxerrBL19[o], rvmaxerrlBL19[o], rvmaxerrrBL19[o] = ccflib.rverr_from_sig(rv, sigBL19[o])


            # ipdb.set_trace()
        # --- End orders loop ---

        if False:
            fig, ax = plt.subplots(3,1, sharex=True, figsize=(8,12))
            ax[0].errorbar(ords, rvmaxZ03, yerr=[rvmaxerrlZ03, rvmaxerrrZ03], fmt='o', label='Z03')
            ax[0].errorbar(ords, rvmaxBL19, yerr=[rvmaxerrlBL19, rvmaxerrrBL19], fmt='o', label='BL19')
            ax[0].set_ylabel('RV [km/s]')
            ax[1].errorbar(ords, rvmaxZ03 - rvmaxBL19, fmt='o', label='Z03 - BL19, std {:.3f}'.format(np.nanstd(rvmaxZ03 - rvmaxBL19)))
            ax[1].set_ylabel('RV diff [km/s]')
            ax[2].plot(ords, rvmaxerrZ03, '.', label='Z03 average')
            ax[2].plot(ords, rvmaxerrlZ03, '+', label='Z03 left')
            ax[2].plot(ords, rvmaxerrrZ03, 'x', label='Z03 right')
            ax[2].plot(ords, rvmaxerrBL19, '.', label='BL19 average')
            ax[2].plot(ords, rvmaxerrlBL19, '+', label='BL19 left')
            ax[2].plot(ords, rvmaxerrrBL19, 'x', label='BL19 right')
            ax[2].set_ylabel('RV uncertainty [km/s]')
            ax[-1].set_xlabel('Order')
            ax[0].set_title(os.path.basename(os.path.splitext(filobs)[0]))
            for a in ax.flatten():
                a.minorticks_on()
                a.legend(fontsize='x-small')
            plt.tight_layout()
            plt.show()
            plt.close()

        # -----------------------
        
        # Coadd

        ccsum = np.nansum(cc, axis=0)
        logLZ03sum = np.nansum(logLZ03, axis=0)
        logLBL19sum = np.nansum(logLBL19, axis=0)

        ccferrsum = np.ones_like(ccsum)*np.nan

        # Compute sigma
        sigZ03sum, pZ03sum, dlogLZ03sum = ccflib.logL2sigma(logLZ03sum, dof=1)
        sigBL19sum, pBL19sum, dlogLBL19sum = ccflib.logL2sigma(logLBL19sum, dof=1)

        # RV from maximum logL
        rvmaxZ03sum = rv[np.argmax(logLZ03sum)]
        rvmaxBL19sum = rv[np.argmax(logLBL19sum)]

        # Compute RV err
        rvmaxerrZ03sum, rvmaxerrlZ03sum, rvmaxerrrZ03sum = ccflib.rverr_from_sig(rv, sigZ03sum)
        rvmaxerrBL19sum, rvmaxerrlBL19sum, rvmaxerrrBL19sum = ccflib.rverr_from_sig(rv, sigBL19sum)

        # -----------------------

        # Plots
        if False:
            # CC
            fig, axsum, axo, axmap = ccflib.plot_ccfo_lines_map(rv, cc, ccsum, ccferrsum, lisord=None, title='{}'.format(obsid))
            plotutils.figout_simple(fig, sv=args.plot_sv, filout=os.path.join(args.dirout, '{}_cc'.format(obsid)), svext=args.plot_ext, sh=args.plot_sh)
            # plt.show(), plt.close()

            # logLZ03
            fig, axsum, axo, axmap = ccflib.plot_ccfo_lines_map(rv, logLZ03, logLZ03sum, ccferrsum, lisord=None, title='{} $\logL$ Z03'.format(obsid), ylabelsum = 'Coadd. $\logL$', ylabelline='Order $\logL$', cblabelmap='$\logL$')
            plotutils.figout_simple(fig, sv=args.plot_sv, filout=os.path.join(args.dirout, '{}_logLZ03'.format(obsid)), svext=args.plot_ext, sh=args.plot_sh)
            # plt.show(), plt.close()

            # logLBL19
            fig, axsum, axo, axmap = ccflib.plot_ccfo_lines_map(rv, logLBL19, logLBL19sum, ccferrsum, lisord=None, title='{} $\logL$ BL19'.format(obsid), ylabelsum = 'Coadd. $\logL$', ylabelline='Order $\logL$', cblabelmap='$\logL$')
            plotutils.figout_simple(fig, sv=args.plot_sv, filout=os.path.join(args.dirout, '{}_logLBL19'.format(obsid)), svext=args.plot_ext, sh=args.plot_sh)
            # plt.show(), plt.close()

            # sigZ03
            fig, axsum, axo, axmap = ccflib.plot_ccfo_lines_map(rv, sigZ03, sigZ03sum, ccferrsum, lisord=None, title='{} $\sigma$ Z03'.format(obsid), ylabelsum = 'Coadd. $\sigma$', ylabelline='Order $\sigma$', cblabelmap='$\sigma$')
            plotutils.figout_simple(fig, sv=args.plot_sv, filout=os.path.join(args.dirout, '{}_sigZ03'.format(obsid)), svext=args.plot_ext, sh=args.plot_sh)
            # plt.show(), plt.close()

            # sigBL19
            fig, axsum, axo, axmap = ccflib.plot_ccfo_lines_map(rv, sigBL19, sigBL19sum, ccferrsum, lisord=None, title='{} $\sigma$ BL19'.format(obsid), ylabelsum = 'Coadd. $\sigma$', ylabelline='Order $\sigma$', cblabelmap='$\sigma$')
            plotutils.figout_simple(fig, sv=args.plot_sv, filout=os.path.join(args.dirout, '{}_sigBL19'.format(obsid)), svext=args.plot_ext, sh=args.plot_sh)
            # plt.show(), plt.close()

        # -----------------------

        # Save data
        filout = os.path.join(args.dirout, os.path.basename(os.path.splitext(filobs)[0]) + '_cclogL.fits')
        ccflib.outfits_cclogLall(
            rv, cc, logLZ03, sigZ03, logLBL19, sigBL19, rvmaxZ03,
            rvmaxerrZ03, rvmaxerrlZ03, rvmaxerrrZ03, 
            rvmaxBL19, rvmaxerrBL19, rvmaxerrlBL19, rvmaxerrrBL19,
            ccsum,
            logLZ03sum, sigZ03sum, rvmaxZ03sum, rvmaxerrZ03sum, rvmaxerrlZ03sum, rvmaxerrrZ03sum, 
            logLBL19sum, sigBL19sum, rvmaxBL19sum, rvmaxerrBL19sum, rvmaxerrlBL19sum, rvmaxerrrBL19sum, 
            header, filout)
        # # Read output example:
        # rv, cc, logLZ03, sigZ03, logLBL19, sigBL19, rvmaxZ03, rvmaxerrZ03, rvmaxerrlZ03, rvmaxerrrZ03,  rvmaxBL19, rvmaxerrBL19, rvmaxerrlBL19, rvmaxerrrBL19, ccsum, logLZ03sum, sigZ03sum, rvmaxZ03sum, rvmaxerrZ03sum, rvmaxerrlZ03sum, rvmaxerrrZ03sum, logLBL19sum, sigBL19sum, rvmaxBL19sum, rvmaxerrBL19sum, rvmaxerrlBL19sum, rvmaxerrrBL19sum, header, rvmaxrec = ccflib.infits_cclogLall(filout)


        # Organize data
        # ------- TODO: Save only final ord-coadded data
        # -------       Add per-order data later
        ccfparsum = {}
        ccfparsum['rvmaxZ03sum'] = rvmaxZ03sum
        ccfparsum['rvmaxerrZ03sum'] = rvmaxerrZ03sum
        ccfparsum['rvmaxerrlZ03sum'] = rvmaxerrlZ03sum
        ccfparsum['rvmaxerrrZ03sum'] = rvmaxerrrZ03sum
        ccfparsum['rvmaxBL19sum'] = rvmaxBL19sum
        ccfparsum['rvmaxerrBL19sum'] = rvmaxerrBL19sum
        ccfparsum['rvmaxerrlBL19sum'] = rvmaxerrlBL19sum
        ccfparsum['rvmaxerrrBL19sum'] = rvmaxerrrBL19sum
        dataccfsumTS[filobs] = ccfparsum

        ccfparo = {}
        ccfparo['rvmaxZ03'] = rvmaxZ03
        ccfparo['rvmaxerrZ03'] = rvmaxerrZ03
        ccfparo['rvmaxerrlZ03'] = rvmaxerrlZ03
        ccfparo['rvmaxerrrZ03'] = rvmaxerrrZ03
        ccfparo['rvmaxBL19'] = rvmaxBL19
        ccfparo['rvmaxerrBL19'] = rvmaxerrBL19
        ccfparo['rvmaxerrlBL19'] = rvmaxerrlBL19
        ccfparo['rvmaxerrrBL19'] = rvmaxerrrBL19
        for p in ['rvmaxZ03', 'rvmaxerrZ03', 'rvmaxerrlZ03', 'rvmaxerrrZ03', 'rvmaxBL19', 'rvmaxerrBL19', 'rvmaxerrlBL19', 'rvmaxerrrBL19']:
            # ccfparo[p] = pd.DataFrame(ccfparo[p]).transpose()
            # rename_cols = {o: '{}o{}'.format(p, o) for o in ccfparo[p].columns}
            # ccfparo[p].rename(rename_cols, axis=1, inplace=True)
            index_p = ['{}o{}'.format(p, o) for o in range(len(ccfparo[p]))]
            ccfparo[p] = pd.DataFrame(ccfparo[p], index=index_p, columns=[filobs])

        # Merge order data from dict of dataframes into single dataframe
        # ccfparo = pd.concat(ccfparo.values(), axis=1)
        ccfparo = pd.concat(ccfparo.values()).transpose()
        # dataccfoTS[filobs] = ccfparo
        dataccfoTS.append(ccfparo)

        # --- End observations loop ---

    ###########################################################################


    # Outputs

    # Save all data (TS and order) in a single file

    # Convert to pandas dataframe
    dataccfsumTS = pd.DataFrame.from_dict(dataccfsumTS, orient='index')
    dataccfoTS = pd.concat(dataccfoTS)

    # Join observations input data and CCF data
    # dataall = pd.concat([dataccfsumTS, dataobs], axis=1, sort=False)
    dataall = pd.concat([dataccfsumTS, dataccfoTS, dataobs], axis=1, sort=False)

    # Change index from path/obs to obs
    dataall['filobs'] = dataall.index
    dataall['obs'] = [os.path.basename(filobs) for filobs in dataall['filobs']]
    dataall.set_index('obs', inplace=True, drop=False)

    # Save
    cols = dataall.columns
    filout = os.path.join(args.dirout, '{}.cclogLpar.dat'.format(args.obj))
    dataall.to_csv(filout, sep=' ', na_rep=np.nan, columns=cols, header=True, index=True, float_format='%0.8f')

    # Quick plot
    if True:
        fig, ax = plt.subplots()
        ax.errorbar(dataall['bjd'], dataall['rvmaxZ03sum'], yerr=dataall['rvmaxerrZ03sum'], linestyle='None', marker='o', label='Z03')
        ax.errorbar(dataall['bjd'], dataall['rvmaxBL19sum'], yerr=dataall['rvmaxerrBL19sum'], linestyle='None', marker='o', label='BL19')
        ax.legend(fontsize='small')
        ax.minorticks_on()
        ax.set_xlabel('BJD [d]')
        ax.set_ylabel('RV [km/s]')
        plt.tight_layout()
        plotutils.figout_simple(fig, sv=args.plot_sv, filout=os.path.join(args.dirout, 'ts_rv_quickcompare'), svext=args.plot_ext, sh=args.plot_sh)
        ipdb.set_trace()




    # ipdb.set_trace()


    ###########################################################################


    return

###############################################################################


if __name__ == "__main__":

    print('Running:', sys.argv)

    main(sys.argv[1:])

    print('End')
