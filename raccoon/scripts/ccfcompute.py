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

# import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    parser.add_argument('inst', choices=['HARPS', 'HARPN', 'CARM_VIS', 'CARM_NIR', 'EXPRES'], help='Instrument.')

    parser.add_argument('--filobs2blaze', help='List of blaze file corresponding to each observation. Format: Column 1) filspec (e2ds), Column 2) filblaze. Full paths. For HARPS/N data. Needed if do not want to use the default from the header. If None, get file names from each observation header.', type=str, default=None)
    # parser.add_argument('--dirblaze', help='Directory containing blaze files. For HARPS/N data.', type=str, default=None)

    parser.add_argument('--expresw', choices=['wavelength', 'bary_wavelength', 'excalibur', 'bary_excalibur'], help='EXPRES wavelength.', default='bary_excalibur')

    # Mask
    parser.add_argument('filmask', help='Path to custom mask file (file with extension `.mas`), or mask ID to use one of the default masks, or (CARM GTO) path to "mask selection file" (file with any other extension that specifies the masks available to choose from). Mask file format: Columns: 0) w (wavelengths), 1) f (weights), separated by whitespaces. Mask-selection file format: Columns: 0) object used to make the mask `objmask`, 1) spectral type of `objmask`, 2) `vsini` of `objmask`, 3) path to mask file (`.mas` extension). There can only be one mask file for each combination of spt-vsini. TODO: Only spt is used to select the mask (not the vsini).', type=str)
    parser.add_argument('--maskformatharp', help='If mask format is w1, w2, f and wavelengths are in air -> it is transformed into w, f and vacuum.', action='store_true')
    parser.add_argument('--maskair', help='If mask wavelengths in air, tranform to vacuum. Not needed if `maskformatharp` is True.', action='store_true')
    parser.add_argument('--objmask', help='Overwrites values from `filmask`.', type=str, default=None)
    parser.add_argument('--sptmask', help='Overwrites values from `filmask`.', type=str, default=None)
    parser.add_argument('--vsinimask', help='Overwrites values from `filmask`.', type=float, default=None)
    # parser.add_argument('--filmaskabserr')

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

    # Output
    parser.add_argument('--dirout', help='Output directory.', default='./ccf_output/', type=str)

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

    # Get mask
    # --------

    # Check if filmask is a valid mask ID
    dictmask = ccflib.listmask_default()[args.inst]
    dictmask_ids = dictmask.keys()

    if args.filmask in dictmask_ids:
        filmask = dictmask[args.filmask]['filmask']
        datamask = dictmask[args.filmask]

    # If not check if its a mask file or a mask-selection file
    else:
        if not os.path.exists(args.filmask): sys.exit('Mask file {} does not exist'.format(args.filmask))

        # Mask file
        if os.path.splitext(args.filmask)[1] == '.mas':
            filmask = args.filmask

            datamask = {
                'filmask': filmask,
                'objmask': args.objmask,
                'sptmask': args.sptmask,
                'vsinimask': args.vsinimask,
            }
            verboseprint('\nMask: ', datamask['filmask'])
        # Mask-selection file
        else:
            verboseprint('\nSelecting mask from list in: {}'.format(args.filmask))
            # Select mask
            maskinfo = ccflib.selectmask_carmenesgto(args.filmask, args.spt, args.vsini, sptdefault='M3.5', vsinidefault=2.0, verbose=True)
            filmask = os.path.expanduser(maskinfo['filmask'])

            datamask = {
                'filmask': filmask,
                'objmask': maskinfo['obj'] if args.objmask is None else args.objmask,
                'sptmask': maskinfo['spt'] if args.sptmask is None else args.sptmask,
                'vsinimask': maskinfo['vsini'] if args.vsinimask is None else args.vsinimask,
            }

    verboseprint('Mask: {}'.format(filmask))
    verboseprint('  Mask obj {}, SpT {}, vsini {}'.format(datamask['objmask'], datamask['sptmask'], datamask['vsinimask']))
    verboseprint('  Target {}, SpT {}, vsini {}'.format(args.obj, args.spt, args.vsini))

    # if not os.path.exists(args.filmask): sys.exit('Mask file {} does not exist'.format(args.filmask))

    # Read mask
    if not args.maskformatharp:
        wm, fm = np.loadtxt(filmask, usecols=[0, 1], unpack=True)
    else:
        wm1, wm2, fm = np.loadtxt(filmask, usecols=[0, 1, 2], unpack=True)
        wm = (wm1 + wm2) / 2.
        wm = spectrumutils.wair2vac(wm)
        args.maskair = False
    # Transform to vacuum
    if args.maskair:
        wm = spectrumutils.wair2vac(wm)
        args.maskair = False

    wm_original, fm_original = wm.copy(), fm.copy()
    nlinoriginal = len(wm)
    verboseprint('  {} lines in mask'.format(nlinoriginal))

    # Cut mask to selected wrange
    if args.wrange is not None:
        verboseprint('  Cut mask to range: {} -- {}'.format(args.wrange[0], args.wrange[1]))
        mask = (wm >= args.wrange[0]) & (wm <= args.wrange[1])
        wm = wm[mask]
        fm = fm[mask]
        nlin = len(wm)
        verboseprint('  {} lines in mask'.format(nlin))

    ###########################################################################


    # CCF test: Determine CCF RV center and range
    # -------------------------------------------

    if args.rvcen is None or args.rvrng is None:

        verboseprint('\nCompute CCF test: determine CCF RV center and range')

        # Spectrum
        filblaze = lisobs2blaze[filobsref] if args.filobs2blaze is not None else None
        w, f, sf, c, header, dataextra = spectrographutils.fitsred_read(filobsref, args.inst, carmnirdiv=carmnirordssplit, harpblaze=True, dirblaze=None, filblaze=filblaze, expresw=args.expresw)
        # nord = len(w)
        # ords = np.arange(0, nord, 1)

        # EXPRES use own mask to remove bad pixels
        if args.inst == 'EXPRES':
            w, f, sf, c = expresutils.apply_expres_masks_spec(w, f, sf, c, dataextra['pixel_mask'], excalibur_mask=dataextra['excalibur_mask'] if args.expresw == 'bary_excalibur' or args.expresw == 'excalibur' else None)

        # Transform spectra to vacuum (only HARPS/N)
        if args.inst == 'HARPS' or args.inst == 'HARPN':
            w = spectrumutils.wair2vac(w)

        # Remove pixels with nan or negative wavelengths
        spec_removenan, masknan = spectrumutils.remove_nan_echelle(w, f, sf, c, ords_use=args.ords_use, returntype='original')
        w, f, sf, c = spec_removenan


        # Correct spectrum wavelength
        shift = dataobs.loc[filobsref]['shift']
        wcorr = [w[o] * (1. - (shift) / C_MS) for o in ords]

        # Correct spectrum cont
        f = [f[o] / c[o] for o in ords]

        # Correct spectrum slope
        if args.inst == 'HARPS' or args.inst == 'HARPN':
            fraw = f.copy()
            for o in ords:
                SpecFitPar = np.polyfit(w[o], f[o], 1)
                f[o] = np.array(f[o] / np.poly1d(SpecFitPar)(w[o]) * np.mean(f[o]))

        # Velocity array
        if args.rvcen is not None: args.ccftestrvcen = args.rvcen
        rvtest = np.arange(args.ccftestrvcen - args.ccftestrvrng, args.ccftestrvcen + args.ccftestrvrng + args.ccftestrvstp, args.ccftestrvstp)

        # Order
        if args.ccftesto is None:
            args.ccftesto = oref

        verboseprint('  Test RV center: {}'.format(args.ccftestrvcen))
        verboseprint('  Test RV range: {}'.format(args.ccftestrvrng))
        verboseprint('  Test RV step: {}'.format(args.ccftestrvstp))
        verboseprint('  Obs: {}\n  Order: {}\n'.format(os.path.basename(filobsref), args.ccftesto))

        # Compute CCF test
        ccftest, _ = ccflib.computeccf(wcorr[args.ccftesto], f[args.ccftesto], c[args.ccftesto], wm, fm, rvtest, ron=None)

        # plt.plot(rvtest, ccftest), plt.show(), plt.close()

        # Plot spectrum+mask and CCF test
        if doplottest:
            fig, ax = plt.subplots(2, 1, figsize=(16, 10))
            # fig, ax = plt.subplots(1,2, figsize=(16, 5), gridspec_kw={'width_ratios': [2,1]})
            # Spec
            args.ccftesto = 45
            fmax = np.nanmax(f[args.ccftesto])
            fn = (f[args.ccftesto]) / fmax
            fmaxn = np.nanmax(fn)  # 1.
            fminn = np.nanmin(fn)
            ax[0].plot(wcorr[args.ccftesto], fn, 'k', label='Spec norm shift')
            # Mask
            ax[0].vlines(wm, fminn, fminn + fm / np.nanmax(fm) * fmaxn, colors='C1', label='Mask norm')
            # Tellurics
            wt, ft = telluricutils.read_mask(args.filtell)
            telluricutils.plot_mask(wt, ft*fmaxn, ax=ax[0], leglab='Telluric mask', alpha=.3, color='k')
            ax[0].set_xlim(np.nanmin(wcorr[args.ccftesto]), np.nanmax(wcorr[args.ccftesto]))
            ax[0].set_ylim(fminn, fmaxn)
            ax[0].set_ylabel('Flux')
            ax[0].set_xlabel('Wavelength')
            ax[0].set_title('{}\norder {}, SNR {:.0f}'.format(os.path.basename(filobsref), args.ccftesto, dataobs['snro{:d}'.format(oref)].loc[filobsref]))
            # CCF
            ccfmax = np.nanmax(ccftest)
            ax[1].plot(rvtest, ccftest/ccfmax, marker='.', label='CCF max {:.2e}'.format(ccfmax))
            ax[1].set_xlabel('RV [km/s]')
            ax[1].set_ylabel('CCF norm')
            for a in ax:
                a.legend(loc='upper right')
            plotutils.figout(fig, filout=os.path.join(args.dirout, 'ccftest_{}_{}'.format(os.path.basename(os.path.splitext(filobsref)[0]), args.ccftesto)), sv=args.plottest_sv, svext=args.plot_ext, sh=args.plottest_sh)

        # ---------------------------------

        # Determine CCF center
        if args.rvcen is None:
            imin = np.nanargmin(ccftest)
            args.rvcen = rvtest[imin]
            verboseprint('  CCF RV cen: {} km/s'.format(args.rvcen))
        else:
            verboseprint('  CCF RV cen fixed by user: {} km/s'.format(args.rvcen))

        # ---------------------------------

        # Determine CCF range by fitting a Gaussian and getting its width
        if args.rvrng is None:

            # Determine fit range: CCF maxima closest to absolute minimum
            # - CCF minima and maxima
            #    Mask nans
            masknan = np.isfinite(ccftest)
            limin, limax1, limax2 = peakutils.find_abspeaks(ccftest[masknan], method='custom')
            # - Maxima closest to CCF minimum
            imin = np.nanargmin(ccftest[masknan])
            i = np.where(limin == imin)[0][0]
            imax1, imax2 = limax1[i], limax2[i]
            # Handle array ends
            if imax2 < len(ccftest[masknan]): imax2p = imax2 + 1  # imax2 plus 1
            else: imax2p = imax2

            # Check that the distance in RV between maxima is at least args.ccftestdmin
            # If not, select next maxima until it happens
            # If never happens, will fit all RV range
            i1, i2 = i, i
            while rvtest[masknan][imax2p] - rvtest[masknan][imax1] < args.ccftestdmin:
                # If reached end of RV array, the fit limits are all the range
                if imax1 == limax1[0] or imax2p == limax2[-1]:
                    imax1 = limax1[0]
                    imax2p = limax2[-1]
                    verboseprint('  Cannot constrain CCF minimum range. Using all range ({} -- {}) to fit a Gaussian and determine width.'.format(rvtest[masknan][imax1], rvtest[masknan][imax2p]))
                    break

                # Go to next closest maxima at each side
                i1 = i1 - 1
                i2 = i2 + 1
                imax1, imax2 = limax1[i1], limax2[i2]
                if imax2 != limax2[-1]: imax2p = imax2 + 1  # imax2 plus 1
                else: imax2p = imax2

            # Fit Gaussian
            x = rvtest[masknan][imax1:imax2p]
            y = ccftest[masknan][imax1:imax2p]
            lmfitresult = peakutils.fit_gaussian_peak(x, y, amp_hint=np.nanmin(y) - np.nanmax(y), cen_hint=rvtest[masknan][imin], wid_hint=1., shift_hint=np.nanmax(y), minmax='min')

            fitpartest = {}
            for p in lmfitresult.params.keys():
                if lmfitresult.params[p].value is not None: fitpartest['fit'+p] = lmfitresult.params[p].value
                else: fitpartest['fit'+p] = np.nan
                if lmfitresult.params[p].stderr is not None: fitpartest['fit'+p+'err'] = lmfitresult.params[p].stderr
                else: fitpartest['fit'+p+'err'] = np.nan
            fitpartest['fwhm'] = peakutils.gaussian_fwhm(wid=fitpartest['fitwid'])
            fitpartest['fitredchi2'] = lmfitresult.redchi

            # Determine CCF width: About 3 * FWHM
            args.rvrng = round(np.ceil(fitpartest['fwhm']*3.))  # [km/s]
            # args.rvrng = round(np.ceil(fitpartest['fwhm']*3.)/5)*5 # [km/s], round to 5
            verboseprint('  RV range: {} km/s'.format(args.rvrng))
        else:
            verboseprint('  RV range fixed by user: {} km/s'.format(args.rvrng))

        # ---------------------------------

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


    # Determine CCF fit function
    # ------------------------------------

    # Function
    if args.fitfunc != 'gaussian':
        sys.exit('Not implemented yet!')

    verboseprint('\nFit:')
    verboseprint('  Function: {}'.format(args.fitfunc))
    verboseprint('  Range: {}'.format(args.fitrng))
    if args.fitrng == 'maxabs' or args.fitrng == 'maxcl':
        if args.fitrngeach: verboseprint('  Find maxima in each observation')
        else: verboseprint('  Use first observation ({}) to fix fit range'.format(os.path.basename(lisfilobs[0])))

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
        wmin = np.nanmin([wspecmin, wm[0], wt_broaden_join[0]]) - 500.
        wmax = np.nanmax([wspecmax, wm[-1], wt_broaden_join[-1]]) + 500.
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
            rvnew = np.arange(rv[0], rv[-1] + (rv[1] - rv[0]), args.bervmax / 2.e3)
        else:
            rvnew = rv

        mask = [[]] * len(rvnew)
        for i, maskshift in enumerate(rvnew):
            wm_shift = wm * (1 + maskshift / C_KMS)

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


    # Compute CCF and params: Observations loop
    # -----------------------------------------

    verboseprint('\nCompute CCF and parameters')

    # --- Start observations loop ---
    # ccf, ccferr, ccfreal, ccferrreal = {}, {}, {}, {}
    # ccfpar = {}
    dataccfsumTS = {}
    first, firsto = True, True
    for i, obs in enumerate(tqdm(lisfilobs)):
        filobs = lisfilobs[i]
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

        # More cleaning...
        # - Flux spikes

        # CARMENES (CARACAL) FOX: Reweight flux so that <counts> = SNR^2
        if args.inst == 'CARM_VIS' or args.inst == 'CARM_NIR' or args.inst == 'EXPRES':
            f = [f[o] * dataobs.loc[filobs]['snro{:d}'.format(o)]**2 / np.mean(f[o]) for o in ords]
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

        # ---------------------------------------------------------------------

        # Orders loop

        # Prepare objects to store data
        ccf, ccferr, ccfreal, ccferrreal = [[]]*nord, [[]]*nord, [[]]*nord, [[]]*nord
        bx, by, bxerr = [[]]*nord, [[]]*nord, [[]]*nord
        ccfpar = {}

        # Use only orders where the mask has lines
        ords_use_lines = [o for o in args.ords_use if len(wmords[o]) > 0]

        # --- Start orders loop ---
        for o in ords_use_lines:

            # Compute CCF
            #  If no nan in spectrum nor in mask, and if mask lines fall on spectrum, this shouldn't fail
            ccfo, ccferro = ccflib.computeccf(wcorr[o], f[o], c[o], wmords[o], fmords[o], rv, ron=ron)
            # ccforeal, ccferroreal = ccflib.computeccf(wcorr[o], f[o], c[o], wmords[o], fmords[o], rvreal, ron=ron, forig=fsnroriginal[o])
            ccforeal, ccferroreal = ccflib.computeccf(wcorr[o], f[o], c[o], wmords[o], fmords[o], rvreal, ron=ron, forig=fsnroriginal if fsnroriginal is None else fsnroriginal[o])

            # # Test
            # if o == 36:
            #     plt.plot(rv, ccfo, '.')
            #     plt.tight_layout()
            #     plt.show()
            #     plt.close()

            # Plot spectrum
            """
            if args.plot_ccfproc:
                print('----------------plotting')
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))

                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[3, 1], height_ratios=[1, 1])
                axspec = fig.add_subplot(gs[0, 0])  # [y, x]
                axspecnorm = fig.add_subplot(gs[1, 0])
                axccf = fig.add_subplot(gs[:, 1])

                axspec.plot(w[o], f[o], linewidth=0.5, marker='.', color='k')

                ylim = axspec.get_ylim()
                axspec.vlines(wmords[o], ylim[0], fmords[o]*0.5*ylim[1], color='C1', alpha=0.8, zorder=10)
                axspec.vlines(wmords[o]*()SHIFT!!!!!!!!!!, ylim[0], fmords[o]*0.5*ylim[1], color='C2', alpha=0.8, zorder=10, linestyle='dashed')
                axspec.vlines(wmords[o]-SHIFT!!!!!!!!!!, ylim[0], fmords[o]*0.5*ylim[1], color='C3', alpha=0.8, zorder=10, linestyle='dashed')
                axspec.set_ylim(ylim)

                axspecnorm.plot(w[o], f[o]/c[o], linewidth=0.5, marker='.', color='k')

                ylim = axspecnorm.get_ylim()
                axspecnorm.vlines(wmords[o], ylim[0], fmords[o]*0.5*ylim[1], color='C1', alpha=0.8, zorder=10)
                axspecnorm.set_ylim(ylim)

                axccf.plot(rv, ccfo, linewidth=0.5, marker='.', color='k')
                # ax[1].plot(w[o], f[o]/c[o], linewidth=0.5, marker='.', color=c)
                plt.tight_layout()
                plotutils.figout(fig, filout=os.path.join(args.dirout, '{}_spec_ccf_o{}'.format(os.path.basename(os.path.splitext(filobs)[0]), o)), sv=args.plot_sv, svext=args.plot_ext, sh=args.plot_sh)
                # plt.show()
                # plt.close()
            """

            # -------------------

            # Fit Gaussian

            # Determine fit range
            # - Fit range for each observation
            if args.fitrngeach:
                imin = np.nanargmin(ccfo)
                ifit1, ifit2 = ccflib.determine_fitrng(args.fitrng, rv, ccfo, imin=imin, verb=args.verbose)
            # - Fit range of the first observation
            else:
                if firsto:
                    imin = np.nanargmin(ccfo)
                    ifit1, ifit2 = ccflib.determine_fitrng(args.fitrng, rv, ccfo, imin=imin, verb=args.verbose)
                    firsto = False

            # Fit
            try:
                x = rv[ifit1:ifit2]
                y = ccfo[ifit1:ifit2]
                lmfitresult = peakutils.fit_gaussian_peak(x, y, amp_hint=np.nanmin(y) - np.nanmax(y), cen_hint=x[np.nanargmin(y)], wid_hint=1., shift_hint=np.nanmax(y), minmax='min')

                paro = {}
                for p in lmfitresult.params.keys():
                    if lmfitresult.params[p].value is not None: paro['fit'+p] = lmfitresult.params[p].value
                    else: paro['fit'+p] = np.nan
                    if lmfitresult.params[p].stderr is not None: paro['fit'+p+'err'] = lmfitresult.params[p].stderr
                    else: paro['fit'+p+'err'] = np.nan
                fwhm, fwhmerr = peakutils.gaussian_fwhmerr(wid=paro['fitwid'], widerr=paro['fitwiderr'])
                paro['fwhm'] = fwhm
                paro['fwhmerr'] = fwhmerr
                paro['fitredchi2'] = lmfitresult.redchi

            except:
                paro = {
                    'fitamp': np.nan, 'fitcen': np.nan, 'fitwid': np.nan, 'fitshift': np.nan, 'fwhm': np.nan,
                    'fitamperr': np.nan, 'fitcenerr': np.nan, 'fitwiderr': np.nan, 'fitshifterr': np.nan, 'fwhmerr': np.nan,
                    'fitredchi2': np.nan,
                }

            # -------------------

            # Compute RV error

            rverrto, dero, rverro = ccflib.computerverr(rv, ccfo/np.nanmean(fmords[o]), ccferro, returnall=True)
            rverrtoreal, deroreal, rverroreal = ccflib.computerverr(rvreal, ccforeal/np.nanmean(fmords[o]), ccferroreal, returnall=True)

            # Add shift error to RV to get "absolute" RV error
            rverrtoabs = np.sqrt(rverrto**2 + (shifterr_use*1.e-3)**2)
            rverrtorealabs = np.sqrt(rverrto**2 + (shifterr_use*1.e-3)**2)

            # Add data
            paro['rv'] = paro['fitcen']
            paro['rverr'] = rverrtoreal  # From real sampling
            paro['rverrabs'] = rverrtorealabs  # From real sampling

            # # Plot RV error
            # if doplottest:
            #     fig, ax = plt.subplots(4,1, sharex=True)
            #     ax[0].plot(rvreal, ccforeal, 'o')
            #     ax[0].set_ylabel('CCF')
            #     ax[1].plot(rvreal, ccferroreal, 'o')
            #     ax[1].set_ylabel('CCFerr')
            #     ax[2].plot(rvreal, dero, 'o')
            #     ax[2].set_ylabel('dCCF/dRV')
            #     ax[3].plot(rvreal, rverro, 'o')
            #     ax[3].set_ylabel('RVerr')
            #     for a in ax:
            #         a.minorticks_on()
            #         # a.legend()
            #     plotutils.figout(fig, filout=os.path.join(args.dirout, 'rverr_{}_o{}'.format(os.path.basename(os.path.splitext(filobsref)[0]), o)), sv=args.plottest_sv, svext=args.plot_ext, sh=args.plottest_sh)

            # -------------------

            # Compute contrast

            contrasto, contrastoerr = peakutils.gaussian_contrasterr(paro['fitamp'], paro['fitshift'], amperr=paro['fitamperr'], shifterr=paro['fitshifterr'])

            # Add data
            paro['contrast'] = contrasto
            paro['contrasterr'] = contrastoerr

            # -------------------

            # Compute bisector

            try:
                bxo, byo, bxoerr, biso, bisoerr = ccflib.computebisector_biserr(rv, ccfo, rverro, n=args.bisectorn, bybotmin_percent=args.bisbotmin, bybotmax_percent=args.bisbotmax, bytopmin_percent=args.bistopmin, bytopmax_percent=args.bistopmax, xrealsampling=args.rvstpreal, verb=False, returnall=False)
            except:
                bxo, byo, bxoerr, biso, bisoerr = [np.nan]*args.bisectorn, [np.nan]*args.bisectorn, [np.nan]*args.bisectorn, np.nan, np.nan

            # Add data
            paro['bis'] = biso
            paro['biserr'] = bisoerr

            # -------------------

            # Organize data
            ccf[o] = ccfo
            ccferr[o] = ccferro
            ccfreal[o] = ccforeal
            ccferrreal[o] = ccferroreal
            ccfpar[o] = paro
            bx[o] = bxo
            by[o] = byo
            bxerr[o] = bxoerr

        # -----------------------

        # Deal with orders with no CCF: Add nans
        ords_empty = [o for o in ords if o not in ords_use_lines]

        for o in ords_empty:
            # Add nan in arrays
            ccf[o] = np.ones_like(rv) * np.nan
            ccferr[o] = np.ones_like(rv) * np.nan
            ccfreal[o] = np.ones_like(rv) * np.nan
            ccferrreal[o] = np.ones_like(rv) * np.nan
            bx[o] = np.ones(args.bisectorn) * np.nan
            by[o] = np.ones(args.bisectorn) * np.nan
            bxerr[o] = np.ones(args.bisectorn) * np.nan

            # Add nan in dict
            ccfpar[o] = {
                'fitamp': np.nan, 'fitcen': np.nan, 'fitwid': np.nan, 'fitshift': np.nan, 'fwhm': np.nan,
                'fitamperr': np.nan, 'fitcenerr': np.nan, 'fitwiderr': np.nan, 'fitshifterr': np.nan, 'fwhmerr': np.nan,
                'fitredchi2': np.nan,
                'rv': np.nan, 'rverr': np.nan,
                'contrast': np.nan, 'contrasterr': np.nan,
                'bis': np.nan, 'biserr': np.nan,
                }

        # --- End orders loop ---

        # Organize orders data
        ccfpar = pd.DataFrame.from_dict(ccfpar, orient='index')
        ccfpar.index.set_names('orders', inplace=True)
        ccfpar.sort_index(inplace=True)

        # Sum CCF
        ccfsum = ccflib.sumccf(rv, ccf, ords_use_lines)
        ccfsumreal = ccflib.sumccf(rvreal, ccfreal, ords_use_lines)

        # if doplottest:
        if False:
            fig, ax = plt.subplots()
            ax.plot(rv, ccfsum, marker='.')
            plt.tight_layout()
            if args.plottest_sh: plt.show()
            plt.close()

        # -----------------------

        # Fit Gaussian

        # Determine fit range
        # - Fit range for each observation
        if args.fitrngeach:
            imin = np.nanargmin(ccfsum)
            ifit1sum, ifit2sum = ccflib.determine_fitrng(args.fitrng, rv, ccfsum, imin=imin, verb=args.verbose)
        # - Fit range of the first observation
        else:
            if first:
                imin = np.nanargmin(ccfsum)
                ifit1sum, ifit2sum = ccflib.determine_fitrng(args.fitrng, rv, ccfsum, imin=imin, verb=args.verbose)
                first = False

        # Fit
        try:
            x = rv[ifit1sum:ifit2sum]
            y = ccfsum[ifit1sum:ifit2sum]
            lmfitresult = peakutils.fit_gaussian_peak(x, y, amp_hint=np.nanmin(y) - np.nanmax(y), cen_hint=x[np.nanargmin(y)], wid_hint=1., shift_hint=np.nanmax(y), minmax='min')
            # Old
            # fitpar = ccflib.fitgaussianfortran(rv, ccfsum, 'maxabs', funcfitnam='gaussian', nfitpar=4)

            # Add data
            ccfparsum = {}
            for p in lmfitresult.params.keys():
                if lmfitresult.params[p].value is not None: ccfparsum['fit'+p] = lmfitresult.params[p].value
                else: ccfparsum['fit'+p] = np.nan
                if lmfitresult.params[p].stderr is not None: ccfparsum['fit'+p+'err'] = lmfitresult.params[p].stderr
                else: ccfparsum['fit'+p+'err'] = np.nan
            ccfparsum['fitredchi2'] = lmfitresult.redchi

        except:
            ccfparsum = {
                'fitamp': np.nan, 'fitcen': np.nan, 'fitwid': np.nan, 'fitshift': np.nan,
                'fitamperr': np.nan, 'fitcenerr': np.nan, 'fitwiderr': np.nan, 'fitshifterr': np.nan,
                'fitredchi2': np.nan,
            }

        # -----------------------

        # Compute RV error

        # - Compute flux error of CCF sum from the flux error of the CCF of each order
        # \sig_{CCFsum}^2(RV) = \sum_{o} \sig_CCFo^2(RV)
        ccfsumerr = np.zeros_like(rv)
        ccfsumrealerr = np.zeros_like(rvreal)
        for o in ords_use_lines:
            ccfsumerr += ccferr[o]**2  # for each RV point
            ccfsumrealerr += ccferrreal[o]**2  # for each RV point
        ccfsumerr = np.sqrt(ccfsumerr)
        ccfsumrealerr = np.sqrt(ccfsumrealerr)

        # - Compute RV error of CCF sum
        # -- Oversampled -> Needed for bisector error
        rverrtsum, dersum, rverrsum = ccflib.computerverr(rv, ccfsum/np.nanmean(fmall_use), ccfsumerr, returnall=True)
        # -- Real sampling -> Needed for real RV error
        rverrtsumreal, dersumreal, rverrsumreal = ccflib.computerverr(rvreal, ccfsumreal/np.nanmean(fmall_use), ccfsumrealerr, returnall=True)

        # Add shift error to RV to get "absolute" RV error
        rverrtsumabs = np.sqrt(rverrtsum**2 + (shifterr_use*1.e-3)**2)
        rverrtsumrealabs = np.sqrt(rverrtsumreal**2 + (shifterr_use*1.e-3)**2)

        # Add data
        ccfparsum['rv'] = ccfparsum['fitcen']
        ccfparsum['rverr'] = rverrtsumreal  # From real sampling
        ccfparsum['rverrabs'] = rverrtsumrealabs  # From real sampling

        # -----------------------

        # Compute FWHM
        #  fwhm = 2 * sqrt(2*ln(2)) * wid
        #  fwhmerr = 2 * sqrt(2*ln(2)) * widerr
        fwhm, fwhmerr = peakutils.gaussian_fwhmerr(wid=ccfparsum['fitwid'], widerr=ccfparsum['fitwiderr'])

        # Add data
        ccfparsum['fwhm'] = fwhm
        ccfparsum['fwhmerr'] = fwhmerr

        # -----------------------

        # Compute contrast
        #   contrast = - (amp/shift) * 100
        #   contrasterr = 100/shift**2 * sqrt( (shift*amperr)**2  + (ampshifterr)**2 )
        contrast, contrasterr = peakutils.gaussian_contrasterr(ccfparsum['fitamp'], ccfparsum['fitshift'], amperr=ccfparsum['fitamperr'], shifterr=ccfparsum['fitshifterr'])

        # Add data
        ccfparsum['contrast'] = contrast
        ccfparsum['contrasterr'] = contrasterr

        # -----------------------

        # Compute bisector
        # "Absolute" RV error not taken into account: use rverrsum instead of rverrsumabs
        try:
            bxsum, bysum, bxsumerr, bissum, bissumerr = ccflib.computebisector_biserr(rv, ccfsum, rverrsum, n=100, bybotmin_percent=args.bisbotmin, bybotmax_percent=args.bisbotmax, bytopmin_percent=args.bistopmin, bytopmax_percent=args.bistopmax, xrealsampling=args.rvstpreal, verb=False, returnall=False)
        except:
            bxsum, bysum, bxsumerr, bissum, bissumerr = [np.nan]*args.bisectorn, [np.nan]*args.bisectorn, [np.nan]*args.bisectorn, np.nan, np.nan

        # Add data
        ccfparsum['bis'] = bissum
        ccfparsum['biserr'] = bissumerr

        # -----------------------

        # Add shift (RV correction) data
        ccfparsum['berv'] = dataobs.loc[filobs]['berv']
        ccfparsum['drift'] = dataobs.loc[filobs]['drift']
        ccfparsum['sa'] = dataobs.loc[filobs]['sa']
        ccfparsum['berverr'] = dataobs.loc[filobs]['berverr']
        ccfparsum['drifterr'] = dataobs.loc[filobs]['drifterr']
        ccfparsum['saerr'] = dataobs.loc[filobs]['saerr']
        ccfparsum['shift'] = dataobs.loc[filobs]['shift']
        ccfparsum['shifterr'] = dataobs.loc[filobs]['shifterr']
        ccfparsum['otherdrift'] = dataobs.loc[filobs]['otherdrift']
        ccfparsum['otherdrifterr'] = dataobs.loc[filobs]['otherdrifterr']

        # Add other observation data
        ccfparsum['bjd'] = dataobs.loc[filobs]['bjd']
        ccfparsum['oref'] = oref
        ccfparsum['snroref'] = dataobs.loc[filobs]['snro{}'.format(oref)]
        ccfparsum['ron'] = dataobs.loc[filobs]['ron']
        ccfparsum['exptime'] = dataobs.loc[filobs]['exptime']
        ccfparsum['airmass'] = dataobs.loc[filobs]['airmass']

        # Add mask and mask number of lines
        ccfparsum['filmask'] = datamask['filmask']
        ccfparsum['filmaskname'] = os.path.basename(datamask['filmask'])
        ccfparsum['objmask'] = datamask['objmask']
        ccfparsum['sptmask'] = datamask['sptmask']
        ccfparsum['vsinimask'] = datamask['vsinimask']
        for o in ords:
            ccfparsum['nlino{}'.format(o)] = nlinords[o]
        ccfparsum['nlint'] = nlin_use
        ccfparsum['nlintallords'] = nlin
        ccfparsum['nlinoriginal'] = nlinoriginal

        # -----------------------

        # Organize data
        dataccfsumTS[filobs] = ccfparsum

        # Save CCF data in FITS (one per obs)
        filout = os.path.join(args.dirout, os.path.basename(os.path.splitext(filobs)[0]) + '_ccf.fits')
        ccflib.outfits_ccfall(rv, ccfsum, ccfparsum, ccf, ccfpar, bxsum, bysum, bx, by, header, filout)
        # verboseprint('  CCF data saved in {}'.format(filout))

        # How to read these FITS files: `ccflib.infits_ccfall`
        # Example:
        # rv2, ccfsum2, ccfparsum2, ccf2, ccfpar2, bxsum2, bysum2, bx2, by2, headerobs2 = ccflib.infits_ccfall(filout)

        # Save CCF data txt
        filout = os.path.join(args.dirout, os.path.basename(os.path.splitext(filobs)[0]) + '_ccf.dat')
        ccflib.outdat_ccf(filout, rv, ccfsum)
        # verboseprint('  CCF data saved in {}'.format(filout))

    # --- End observations loop ---

    ###########################################################################


    # Save sum TS data

    # Convert to pandas dataframe
    dataccfsumTS = pd.DataFrame.from_dict(dataccfsumTS, orient='index')

    # Join observations input data and CCF data
    # dataall = pd.concat([dataccfsumTS, dataobs], axis=1, sort=False)
    dataall = dataccfsumTS

    # Change index from path/obs to obs
    dataall['filobs'] = dataall.index
    dataall['obs'] = [os.path.basename(filobs) for filobs in dataall['filobs']]
    dataall.set_index('obs', inplace=True, drop=False)

    # # Save in file: CCF data, some input data ---> NEW Moved below
    # cols = dataall.columns
    # filout = os.path.join(args.dirout, '{}.par.dat'.format(args.obj))
    # dataall.to_csv(filout, sep=' ', na_rep=np.nan, columns=cols, header=True, index=True, float_format='%0.8f')
    # verboseprint('\nCCF TS data saved in {}'.format(filout))

    # Save in file main output: BJD, RV, FWHM, Contrast, BIS and their errors
    cols = ['bjd', 'rv', 'fwhm', 'contrast', 'bis', 'rverrabs', 'fwhmerr', 'contrasterr', 'biserr']
    filout = os.path.join(args.dirout, '{}.ccfpar.dat'.format(args.obj))
    dataall.to_csv(filout, sep=' ', na_rep=np.nan, columns=cols, header=False, index=False, float_format='%0.8f')
    # verboseprint('CCF TS data saved in {}'.format(filout))

    # Save file info.csv
    cols = ['bjd', 'obs', 'berv', 'drift', 'sa', 'rverr', 'drifterr', 'exptime', 'airmass', 'snroref', 'objmask', 'sptmask', 'vsinimask', 'nlinoriginal', 'nlint']
    filout = os.path.join(args.dirout, '{}.info.csv'.format(args.obj))
    dataall.to_csv(filout, sep=';', na_rep=np.nan, columns=cols, header=False, index=False, float_format='%0.8f')
    # verboseprint('CCF extra data saved in {}'.format(filout))

    ###########################################################################

    # Save individual order data (read FITS files)

    # lisccforv, lisccfofwhm, lisccfocontrast, lisccfobis = {}, {}, {}, {}
    # lisccforverr, lisccfofwhmerr, lisccfocontrasterr, lisccfobiserr = {}, {}, {}, {}
    lisparam = ['rv', 'fwhm', 'contrast', 'bis', 'rverr', 'fwhmerr', 'contrasterr', 'biserr']
    dataorder = {'rv': {}, 'fwhm': {}, 'contrast': {}, 'bis': {}, 'rverr': {}, 'fwhmerr': {}, 'contrasterr': {}, 'biserr': {}}
    lisparam_all = ['rv', 'rverr', 'fwhm', 'fwhmerr', 'contrast', 'contrasterr', 'bis', 'biserr', 'fitamp', 'fitamperr', 'fitcen', 'fitcenerr', 'fitwid', 'fitwiderr', 'fitshift', 'fitshifterr', 'fitredchi2']
    dataorder_all = {'rv': {}, 'rverr': {}, 'fwhm': {}, 'fwhmerr': {}, 'contrast': {}, 'contrasterr': {}, 'bis': {}, 'biserr': {}, 'fitamp': {}, 'fitamperr': {}, 'fitcen': {}, 'fitcenerr': {}, 'fitwid': {}, 'fitwiderr': {}, 'fitshift': {}, 'fitshifterr': {}, 'fitredchi2': {}}
    for i, obs in enumerate(lisfilobs):
        # Read CCF order data
        filobs = lisfilobs[i]
        filccf = os.path.join(args.dirout, os.path.basename(os.path.splitext(filobs)[0]) + '_ccf.fits')
        if not os.path.exists(filccf): continue
        rv, ccfsum, ccfparsum, ccf, ccfpar, bxsum, bysum, bx, by, headerobs = ccflib.infits_ccfall(filccf)
        # Organise data main CCF params
        for p in lisparam:
            dataorder[p][ccfparsum['bjd']] = ccfpar[p]
        #  Organise data all CCF params
        for p in lisparam_all:
            # dataorder_all[p][ccfparsum['bjd']] = ccfpar[p]
            dataorder_all[p][os.path.basename(obs)] = ccfpar[p]

    # Join and save data (main params)
    for p in lisparam:
        # Main params
        dataorder[p] = pd.DataFrame.from_dict(dataorder[p], orient='index')
        filout = os.path.join(args.dirout, '{}.{}o.dat'.format(args.obj, p))
        dataorder[p].to_csv(filout, sep=' ', na_rep=np.nan, header=False)

    ###########################################################################

    # Save all data (TS and order) in a single file

    # Join data (all params), as above with main params
    for p in lisparam_all:
        dataorder_all[p] = pd.DataFrame.from_dict(dataorder_all[p], orient='index')
        rename_cols = {o: '{}o{}'.format(p, o) for o in dataorder_all[p].columns}
        dataorder_all[p].rename(rename_cols, axis=1, inplace=True)

    # Merge order data from dict of dataframes into single dataframe
    dataorder_all = pd.concat(dataorder_all.values(), axis=1)

    # Merge order data `dataorder_all` and TS data `dataall`
    datafinal = pd.concat([dataall, dataorder_all], axis=1)

    # Save in file: CCF data (TS and order), and some input data
    cols = datafinal.columns
    filout = os.path.join(args.dirout, '{}.par.dat'.format(args.obj))
    datafinal.to_csv(filout, sep=' ', na_rep=np.nan, columns=cols, header=True, index=True, float_format='%0.8f')
    # verboseprint('\nCCF TS data saved in {}'.format(filout))

    ###########################################################################

    # Plots

    # Load SERVAL data and merge with CCF data
    if args.dirserval is not None:
        # Read SERVAL data
        dataserval = carmenesutils.serval_get(args.dirserval, obj=args.obj, lisdataid=['rvc', 'info'], inst=args.inst)
        # Change index from bjd to obs
        dataserval['bjd'] = dataserval.index
        dataserval.set_index('obs', inplace=True)
        # Merge SERVAL data with CCF data
        dataall = pd.concat([dataall, dataserval[['servalrvc', 'servalrvcerr']]], axis=1, join='inner', sort=True)
    else:
        dataserval = None

    # ---------------------------

    # Plot CCF parameters TS
    # ...

    # ---------------------------

    # Plot CCF parameters TS (and SERVAL rvc if available)

    if dataserval is not None:
        fig, ax = ccflib.plot_ccfparbasic_servalrvc(dataall, title='{} CCF parameters  {} obs'.format(args.obj, len(dataall.index)))
        plotutils.figout(fig, filout=os.path.join(args.dirout, '{}.ccfpar_servalrvc'.format(args.obj)), sv=args.plot_sv, svext=args.plot_ext, sh=args.plot_sh)

        fig, ax = ccflib.plot_ccfparbasic_servalrvc_separated(dataall, dataserval=dataserval, title='{} CCF parameters'.format(args.obj))
        plotutils.figout(fig, filout=os.path.join(args.dirout, '{}.ccfpar_servalrvc_separated'.format(args.obj)), sv=args.plot_sv, svext=args.plot_ext, sh=args.plot_sh)

    else:
        fig, ax = ccflib.plot_ccfparbasic_servalrvc(dataall, plotserval=False, title='{} CCF parameters  {} obs'.format(args.obj, len(dataall.index)))
        plotutils.figout(fig, filout=os.path.join(args.dirout, '{}.ccfpar'.format(args.obj)), sv=args.plot_sv, svext=args.plot_ext, sh=args.plot_sh)

    # ---------------------------

    # Plot RV CCF and SERVAL, and difference
    if dataserval is not None:
        fig, ax = ccflib.plot_ccfrv_servalrvc_diff(dataall, shiftserval=True, title='{}  {} obs'.format(args.obj, len(dataall.index)))
        plotutils.figout(fig, filout=os.path.join(args.dirout, '{}.ccfrv_servalrvc_diff'.format(args.obj)), sv=args.plot_sv, svext=args.plot_ext, sh=args.plot_sh)

    return

###############################################################################


if __name__ == "__main__":

    print('Running:', sys.argv)

    main(sys.argv[1:])

    print('End')
