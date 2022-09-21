#!/usr/bin/env python
"""
Build a mask from a spectral template.
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import textwrap
# import time
import ipdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

from raccoon import ccf as ccflib
from raccoon import carmenesutils
from raccoon import espressoutils
from raccoon import peakutils
from raccoon import phoenixutils
from raccoon import plotutils
from raccoon import pyutils
from raccoon import spectrographutils
from raccoon import spectrumutils
from raccoon import telluricutils


# Plots
mpl.rcdefaults()
plotutils.mpl_custom_basic()
plotutils.mpl_size_same()

# Constants
C_MS = 2.99792458*1.e8  # Light speed [m/s]
C_KMS = 2.99792458*1.e5  # Light speed [km/s]

# Strings
angstromstr = r'$\mathrm{\AA}$'
angstromstrunit = r'$[\mathrm{\AA}]$'

###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
        `mask.py`

        Construct a weigthed binary mask from selected lines in a spectrum template.

        Arguments
        ---------
        '''),
        epilog=textwrap.dedent('''
            '''),
        formatter_class=pyutils.CustomFormatter)

    # Template
    parser.add_argument('filtpl', help='File containing the spectrum template to be used to construct the mask', type=str, default=None)
    parser.add_argument('tpltype', choices=['serval', 'phoenix', '1dtxt', 'espressos1dcoadd'], help='', type=str)  # 'custommatrix', 'customstepctn'

    parser.add_argument('obj', help='ID', type=str)

    parser.add_argument('--inst', choices=['CARM_VIS', 'CARM_NIR', 'HARPS', 'HARPN', 'ESPRESSO'])

    # Mask shift to RV 0
    # Shift mask by `--tplrv`.
    # Additionally can refine the RV by computing the CCF of the mask with a PHOENIX spectrum broadened
    parser.add_argument('--tplrv', help='Template RV. Options: a) float [km/s], b) carmencita. If None (default), the mask will be shifted 0 km/s. If carmencita data in NaN, the shift will be 0.', default=None, type=str)

    # CARMENES data
    parser.add_argument('--dircarmencita', help='Absolute path.', default=None)
    parser.add_argument('--carmencitaversion', help='', default=None)

    # PHOENIX
    parser.add_argument('--filphoenixw', help='Path to PHOENIX wavelegth file, or nothing, so that the provided file (`../data/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits`) is used.', nargs='?', const='../data/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', default=None)
    parser.add_argument('--filphoenixf', help='Path to PHOENIX flux file, or nothing, so that a file will be chosen from the provided ones (using carmencita information).', nargs='?', const='select', default=None)
    parser.add_argument('--phoenixbroadinstR', help='Resolution of the instrument to apply instrumental broadening to PHOENIX. If only option present (no R value), the default value for each instrument INST is used.', nargs='?', const='inst', type=str, default=None)
    # parser.add_argument('--phoenixbroadrotV', help='Rotational velocity to apply to PHOENIX [km/s]. Float or carmencita (vsini of OBJ).', nargs='?', const='carmencita', type=str, default=None)
    parser.add_argument('--phoenixwmin', help='', default=None, type=float)
    parser.add_argument('--phoenixwmax', help='', default=None, type=float)

    # Telluric mask
    parser.add_argument('--filtell', help='File containing a telluric mask. If None, no tellurics are removed', type=str, default=None)
    parser.add_argument('--tellbroadendv', help='Velocity by which to broaden the telluric lines to be removed. Options: A) float in m/s (default is 30000), B) `obsall` (default): maximum BERV of all observations available. If B), must provide a list of the observations used to make the template used with `--lisfilobs`.', type=str, default='obsall')  # nargs='+',
    parser.add_argument('--lisfilobs', help='See `tellbroadendv`. File with the names of the reduced FITS spectra or directly the file names (names must include the absolute path to the files). The file with the list cannot end in `.fits`.', nargs='+', type=str)

    # Lines and regions to remove
    # TODO implement these removal
    # IMPORTANT: regions given in what wavelength reference? Take into account RV of the star
    # parser.add_argument('--fillinermv', help='File with positions and width of lines to remove. Columns: (0) wcen [A], (1) width [km/s]. If None, no lines are removed.', type=str, default=None)
    # parser.add_argument('--filregionrmv', help='File with positions of regions to remove (emission lines, bands). Columns: (0) w1 [A], (1) w2 [A]. If None, no lines are removed.', type=str, default=None)

    parser.add_argument('--rmvpixblue', help='Cut spectrum in first blue pixels of first orders (because noisy). CARM_VIS.', action='store_true')
    parser.add_argument('--rmvpixblue_npix', help='Number of pixels to remove at the begining of the order.', type=int, default=500)
    parser.add_argument('--rmvpixblue_ords', help='Orders to apply the blue pixel removal.', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    parser.add_argument('--ords_use', nargs='+', help='Orders to consider for the mask. Orders depend on the instrument the template comes from! The orders are counted from 0 to N-1, where N is the number of orders in the template file - so these are not the real orders. Only applies if `--tpltype serval`. If None (default), all orders are used.', default=None)

    parser.add_argument('--wmin', help='Minimum wavelength of the mask, in A. If None, use minimum wavelength of the template provided.', type=float, default=None)
    parser.add_argument('--wmax', help='Maximum wavelength of the mask, in A. If None, use maximum wavelength of the template provided.', type=float, default=None)

    # Template smoothing
    parser.add_argument('--smoothkernel', choices=['gaussian', 'box'], help='Kernel to convolve with the template spectrum to smooth it.', type=str, default=None)
    parser.add_argument('--smoothkernelwidth', help='Width of the smoothing kernel. If kernel is `box`, KERNELWIDTH is the width of the filter, and if kernel is `gaussian`, it is the standard deviation of the gaussian.', type=int, default=2)

    # Template continuum fitting and normalization
    parser.add_argument('--cont', help='Funtion to fit to the spectrum template continuum in order to normalize it.', type=str, choices=['poly', 'spl'], default=None)

    parser.add_argument('--cont_linesmaskstar', help='Ignore certain stellar strong lines when normalizing. The line positions depend on the RV of the star, corrected using --tplrv.', action='store_true')

    parser.add_argument('--cont_linesmasktell', help='Ignore certain strong telluric bands when normalizing.', action='store_true')
    parser.add_argument('--cont_linesmasktell_read', help='', type=str, default=None)

    parser.add_argument('--cont_pixblue', help='Ignore first blue pixels of first orders when normalizing. CARM_VIS. Not needed if `--rmvpixblue`, because the noisy pixels are already removed from the spectrum. ', action='store_true')
    parser.add_argument('--cont_pixblue_npix', help='Number of pixel to ignore at the begining of each order.', type=int, default=150)
    parser.add_argument('--cont_pixblue_ords', help='Orders to apply the blue pixel removal.', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    parser.add_argument('--contfiltmed', help='Size of the median filter to smooth out single-pixel deviations when fitting the template continuum. Must be an odd integer.', type=int, default=1)
    parser.add_argument('--contfiltmax', help='Size of the maximum filter to ignore deeper fluxes of absorption lines when fitting the template continuum', type=int, default=400)

    parser.add_argument('--contpolyord', help='Order of the polynomial function to be fit to the template spectrum. Applies only if `--cont poly`.', type=int, default=2)
    parser.add_argument('--contsplsmooth', help='Positive smoothing factor used to choose the number of knots. Number of knots will be increased until the smoothing condition is satisfied: sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s. If None, s=len(w), where w=weights (should be a good value if 1/w[i] is an estimate of the standard deviation of y[i]). If 0, spline will interpolate through all data points. Only applies if `--cont spl`.', type=int, default=None)
    parser.add_argument('--contspldegree', help='Degree of the smoothing spline. Default: 3, cubic spline.', type=int, default=3)

    # Template minima
    parser.add_argument('--filfit', help='Directory containing the file(s) from which to read the (previously generated) minima fit results. Directory will contain a single file if the template is a single spectrum (`tpltype` is phoenix or other), or directory will the fit minima results of different orders if template has several orders (`tpltype` is serval). If None, fit minima are computed.', type=str, default=None)

    # Minima selection
    parser.add_argument('--line_contrast1min', help='', type=float, default=0)
    parser.add_argument('--line_contrast1max', help='', type=float, default=1)
    parser.add_argument('--line_contrast2min', help='', type=float, default=0)
    parser.add_argument('--line_contrast2max', help='', type=float, default=1)
    parser.add_argument('--line_contrastminmin', help='', type=float, default=0)
    parser.add_argument('--line_contrastminmax', help='', type=float, default=1)
    parser.add_argument('--line_contrastmaxmin', help='', type=float, default=0)
    parser.add_argument('--line_contrastmaxmax', help='', type=float, default=1)
    parser.add_argument('--line_contrastmeanmin', help='', type=float, default=0)
    parser.add_argument('--line_contrastmeanmax', help='', type=float, default=1)
    parser.add_argument('--line_fwhmmin', help='[km/s]', type=float, default=0)
    parser.add_argument('--line_fwhmmax', help='[km/s]', type=float, default=9999)
    parser.add_argument('--line_depthmin', help='', type=float, default=0)
    parser.add_argument('--line_depthmax', help='', type=float, default=1.5)
    parser.add_argument('--line_depthw_type', help='', choices=['constant', 'poly'], default='constant')
    parser.add_argument('--line_depthw_percentdeepest', help='', type=float, default=0)
    parser.add_argument('--line_depthw_depthmaxquantile', help='', type=float, default=1)
    parser.add_argument('--line_depthw_polyord', help='', type=int, default=2)

    parser.add_argument('--maskwave', help='', type=str, default='cen')
    parser.add_argument('--maskweight', help='', type=str, default='contrastmeanfwhm-1')

    parser.add_argument('--condnodepth', help='Do not take depth conditions into account when selection minima (for non-normalised templates', action='store_true')

    # Orders join
    parser.add_argument('--dwmin', help='Minimum separation between to lines to be considered different lines [A]', type=float, default=0.05)
    parser.add_argument('--joinord', choices=['keepred', 'merge'], help='How to join the orders. If `keepred`, keep the redder order lines in the overlap region (bluer part of the orders more noisy), if `merge`, merge blue and red extremes of consecutive orders', default='merge')

    # Outputs
    parser.add_argument('--dirout', help='General output directory. Absolute path.', type=str, default='./mask_output/')
    parser.add_argument('--dirout_minfit', help='Output directory of the minima fit results, inside DIROUT (DIROUT/DIROUT_MINFIT/) ', type=str, default='minfit/')
    parser.add_argument('--dirout_tpl', help='Output directory of the processed template, inside DIROUT (DIROUT/DIROUT_TPL/) ', type=str, default='./')
    parser.add_argument('--dirout_mask', help='Output directory of the mask, inside DIROUT (DIROUT/DIROUT_MASK/) ', type=str, default='./')
    parser.add_argument('--dirout_plot', help='Directory inside DIROUT where to store general plots (DIROUT/DIROUT_PLOTS/)', type=str, default='plots/')

    parser.add_argument('--filoutmask', help='Name of the mask file. Options: a) None (default): name of the input spectrum template changing extension to `.mas`, b) `cuts`: some of the values of the parameters used to select the lines are added to the name, c) any other name.', type=str, default=None)

    parser.add_argument('--verbose', help='', action='store_true')

    # Plots
    parser.add_argument('--plot_sv', help='Make and save plots.', action='store_true')
    parser.add_argument('--plot_sh', help='Show all plots.', action='store_true')
    parser.add_argument('--plot_test', help='Make and show testing plots.', action='store_true')
    parser.add_argument('--plot_finalslice', help='Plot final mask and tpl.', action='store_true')
    parser.add_argument('--plot_ext', nargs='+', help='Extensions of the plots to be saved (e.g. `--plot_ext pdf png`)', default='pdf')

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # Verbosity
    verboseprint = print if args.verbose else lambda *a, **k: None

    verboseprint('\n')
    verboseprint('#'*40)
    verboseprint('\nCompute mask\n')
    verboseprint('#'*40)
    verboseprint('\n')

    verboseprint('Dirout:', args.dirout)

    # Extend directories and files
    if isinstance(args.filtpl, str): args.filtpl = os.path.expanduser(args.filtpl)
    if isinstance(args.dirout, str): args.dirout = os.path.expanduser(args.dirout)
    if isinstance(args.dircarmencita, str): args.dircarmencita = os.path.expanduser(args.dircarmencita)
    if isinstance(args.filphoenixw, str): args.filphoenixw = os.path.expanduser(args.filphoenixw)
    if isinstance(args.filphoenixf, str): args.filphoenixf = os.path.expanduser(args.filphoenixf)
    if isinstance(args.filtell, str): args.filtell = os.path.expanduser(args.filtell)
    if isinstance(args.lisfilobs, str): args.lisfilobs = os.path.expanduser(args.lisfilobs)
    if isinstance(args.cont_linesmasktell_read, str): args.cont_linesmasktell_read = os.path.expanduser(args.cont_linesmasktell_read)
    if isinstance(args.filfit, str): args.filfit = os.path.expanduser(args.filfit)

    # Join directories
    args.dirout_mask = os.path.join(args.dirout, args.dirout_mask)
    args.dirout_tpl = os.path.join(args.dirout, args.dirout_tpl)
    args.dirout_minfit = os.path.join(args.dirout, args.dirout_minfit)
    args.dirout_plot = os.path.join(args.dirout, args.dirout_plot)

    # Create output directories if they do not exist
    for d in [args.dirout, args.dirout_minfit, args.dirout_mask, args.dirout_tpl, args.dirout_plot]:
        if not os.path.exists(d): os.makedirs(d)

    # Make sure figure extensions is a list
    if not isinstance(args.plot_ext, list): args.plot_ext = [args.plot_ext]

    # # Make sure tellbroadendv is a list
    # if not isinstance(args.tellbroadendv, list): args.tellbroadendv = [args.tellbroadendv]

    # Load Carmencita
    if args.dircarmencita is not None: datacarmencita = carmenesutils.carmencita_get(args.dircarmencita, version=args.carmencitaversion, verbose=args.verbose)

    # Make sure tplrv is `carmencita` or a valid float, and get it
    if args.tplrv is not None:
        if args.tplrv == 'carmencita':
            # Find RV of the object in Carmencita
            tplrv = datacarmencita['Vr_kms-1'].loc[args.obj]  # [km/s]
            if not np.isfinite(tplrv):
                tplrv = 0.
                verboseprint('tplrv from carmecita not finite: {}. Setting it to 0.'.format(tplrv))
                # sys.exit('tplrv from carmecita not finite: {}'.format(tplrv))
        elif pyutils.isfloatnum(args.tplrv):
            tplrv = float(args.tplrv)
        else:
            sys.exit('args.tplrv = {}, not valid. Exit.'.format(args.tplrv))
    else:
        tplrv = 0.

    # Save current command
    with open(os.path.join(args.dirout, 'cmdlast.txt'), 'w') as fout:
        fout.write(' '.join(sys.argv))
    with open(os.path.join(args.dirout, 'cmdhistory.txt'), 'a') as fout:
        fout.write(' '.join(sys.argv))
        fout.write('\n')
    # Save command line arguments
    with open(os.path.join(args.dirout, 'cmdlastargs.txt'), 'w') as fout:
        for k, v in sorted(args.__dict__.items()):
            fout.write('{}: {}\n'.format(k, v))

    ###############################################################################

    # Spectrum template
    # -----------------

    # Read spectrum template
    # If template made from observations which contain tellurics, make sure spectrum has not been shifted, i.e. tellurics at 0

    verboseprint('\nRead spectrum template {} ({})'.format(args.filtpl, args.tpltype))

    if args.tpltype == 'serval':
        w, f, sf, header = carmenesutils.serval_tpl_read(args.filtpl)
        nord = len(w)
        ords = np.arange(0, nord, 1)

    elif args.tpltype == 'phoenix':
        sys.exit('Template type is phoenix. Not implemented yet!')
        # TO DO: in parser, specify option for phoenix files
        # w, f = phoenixutils.read_phoenixfits(filw, filf)
        nord = 1
        ords = np.array([0])

    elif args.tpltype == 'espressos1dcoadd':
        w, f, ferr, snr, q, contrib = espressoutils.drs_fitsred_s1dcoadd_read(args.filtpl)
        w, f = [w], [f]
        nord = 1
        ords = np.array([0])
        if False:
            fig, ax = plt.subplots(4,1,figsize=(16,8), sharex=True)
            # ax[0].errorbar(w, f, yerr=ferr, linestyle='None', marker='o', color='0.2')
            ax[0].plot(w, f, color='0.2')
            ax[0].set_ylabel('Flux')
            ax[1].plot(w, ferr, color='0.2')
            ax[1].set_ylabel('Flux error')
            ax[2].plot(w, snr)
            ax[2].set_ylabel('S/N')
            ax[3].plot(w, contrib)
            ax[3].set_ylabel('# input spec.')
            ax[-1].set_xlabel('Wavelength $[\AA]$')
            for a in ax:
                a.minorticks_on()
            plt.tight_layout()
            plt.show(), plt.close()

    elif args.tpltype == '1dtxt':
        w, f = np.loadtxt(args.filtpl, unpack=True)
        # sys.exit('Template type is other. Not implemented yet!')
        # pass
        w, f = [w], [f]
        nord = 1
        ords = np.array([0])

        # ###### TEST
        # w, f = [w[0][760000:810000]], [f[0][760000:810000]]


    if args.ords_use is None: args.ords_use = ords
    else: args.ords_use = np.asarray(args.ords_use, dtype=int)

    try:
        wraw = w.copy()
        fraw = f.copy()
    except:
        wraw = [i.copy() for i in w]
        fraw = [i.copy() for i in f]

    # Remove bad orders
    # -----------------

    # If order empty -> Do not use, remove from args.ords_use
    oempty = [o for o in args.ords_use if len(w[o]) == 0]
    # maskoempty = ~np.asarray([len(w[o])==0 if o in args.ords_use else False for o in ords])
    # args.ords_use = args.ords_use[maskoempty]

    # If wavelength or flux of an order is all nan -> Do not use, remove from args.ords_use
    ownan = [o for o in args.ords_use if ~np.isfinite(w[o]).all()]
    ofnan = [o for o in args.ords_use if ~np.isfinite(f[o]).all()]

    # If wavelength or flux of an order is all 0 -> Do not use, remove from args.ords_use
    ow0 = [o for o in args.ords_use if not np.any(w[o])]
    of0 = [o for o in args.ords_use if not np.any(f[o])]

    # Total orders to remove
    ormv = np.sort(np.unique(oempty + ownan + ofnan + ow0 + of0))
    ords_use_new = []
    for o in args.ords_use:
        if o in ormv:
            continue
        else:
            ords_use_new.append(o)
    args.ords_use = np.asarray(ords_use_new, dtype=int)

    # Remove noisy blue pixels
    # ------------------------

    # Cut orders instead of removing the lines in noisy parts in the mask because of order overlap.

    if args.rmvpixblue:
        wnew, fnew = [[]]*nord, [[]]*nord
        for o in ords:
            if o in args.rmvpixblue_ords:
                # Find wavelength closest to pixel
                wcut = w[o][args.rmvpixblue_npix]
                # Mask
                mask = w[o] >= wcut
                wnew[o] = w[o][mask]
                fnew[o] = f[o][mask]
            else:
                wnew[o] = w[o]
                fnew[o] = f[o]
        w = np.array(wnew)
        f = np.array(fnew)

    # sys.exit()

    ###############################################################################

    # Telluric mask

    if args.filtell is not None:
        verboseprint('\nRead telluric mask', args.filtell)

        # Tellurics
        wt, ft = telluricutils.read_mask(args.filtell)
        wt1, wt2 = telluricutils.mask2wlimits(wt, ft)

        # Get broaden velocity
        if args.tellbroadendv == 'obsall':

            # Get list of observations used to create the template
            # Check if input is file with list or directly the filenames
            if args.lisfilobs is None:
                sys.exit('Must provide `lisfilobs` if tellbroadendv == obsall!')
            # - more than 1 FITS filename in input
            if len(args.lisfilobs) > 1:
                # args.lisfilobs = [os.path.expanduser(i) for i in args.lisfilobs]
                args.lisfilobs = args.lisfilobs
            # - single FITS filename in input
            elif os.path.splitext(args.lisfilobs[0])[1] == '.fits':
                # args.lisfilobs = [os.path.expanduser(i) for i in args.lisfilobs]
                args.lisfilobs = args.lisfilobs
            # - file with list in input
            else:
                # Read names of the files to be downloaded
                args.lisfilobs = os.path.expanduser(args.lisfilobs[0])
                args.lisfilobs = np.loadtxt(args.lisfilobs, dtype='str', usecols=[0])
            # Get observations shift
            # ---- TODO Change to a simpler function that ONLY gets the BERV from the observations header
            _, _, lisshift = spectrographutils.serval_header_rvcorrection_lisobs(args.lisfilobs, args.inst, source='header', obj=args.obj, notfound=np.nan, ext=0)
            lisberv = lisshift['berv']

            # If serval template, remove observations not used in creation
            if args.tpltype == 'serval':
                try:
                    nobstpl = header['HIERARCH SERVAL COADD NUM']
                    lisfiltpl = [header['HIERARCH COADD FILE {:03d}'.format(i+1)] for i in range(nobstpl)]
                    mask = [i in lisfiltpl for i in lisberv.index]
                    if len(lisberv[mask]) > 0:
                        lisberv = lisberv[mask]
                except:
                    pass

            bervmax = np.nanmax(np.abs(lisberv))
            verboseprint('  Maximum BERV {}'.format(bervmax))
            tellbroadendv = bervmax

        elif pyutils.isfloatnum(args.tellbroadendv):
            tellbroadendv = float(args.tellbroadendv)
        else:
            sys.exit('{} tellbroadendv not valid'.format(args.tellbroadendv))
        verboseprint('  Broaden telluric features by +-{} m/s'.format(tellbroadendv))

        # Broaden
        wt_broaden = telluricutils.broaden_mask(wt, ft, tellbroadendv)
        wt1_broaden, wt2_broaden = telluricutils.broaden_wlimits(wt1, wt2, tellbroadendv)

        # Join overlaping lines
        wt1_broaden_join, wt2_broaden_join = telluricutils.join_overlap_wlimits(wt1_broaden, wt2_broaden)
        # Make mask with joined limits
        wt_broaden_join, ft_broaden_join = telluricutils.wlimits2mask(wt1_broaden_join, wt2_broaden_join, dw=0.001)

    else:
        wt, ft, wt1, wt2, wt_broaden, wt1_broaden, wt2_broaden, wt1_broaden_join, wt2_broaden_join, wt_broaden_join, ft_broaden_join = [], [], [], [], [], [], [], [], [], [], []

    ###############################################################################

    # Emission lines and other bad regions

    # ...

    ###############################################################################

    # Regions to be masked (i.e. no lines in these wavelengths): tellurics + emission + bad regions
    # TO DO

    wmaskbad = wt_broaden_join
    fmaskbad = ft_broaden_join

    # Extend mask (with flux 0) to all the spectrum range, so there are no interpolation problems
    wspecmin, wspecmax = np.nanmin(np.concatenate(w)), np.nanmax(np.concatenate(w))
    wmaskbad = np.concatenate(([wspecmin], wmaskbad, [wspecmax]))
    fmaskbad = np.concatenate(([0], fmaskbad, [0]))

    # Original masks: 0 good regions, 1 regions to be masked
    # Invert masks: 0 to bad, 1 to good
    Maskbad_inv, fmaskbad_inv = telluricutils.interp_mask_inverse(wmaskbad, fmaskbad, kind='linear')

    # Plot raw spectrum ...

    # -----------------------------------------------------------------------------

    # Smooth spectrum template

    if args.smoothkernel is not None:
        verboseprint('\nSmooth template')
        fconv = spectrumutils.conv_echelle(f, args.smoothkernel, args.smoothkernelwidth, boundary='extend', ords_use=args.ords_use, returnfill=True)
        f = [i.copy() for i in fconv]

        # if args.plot_sh_smooth:
        #     # Plot...
        #     # TO DO
        #     pass

        if args.plot_test:
            fig, ax = plt.subplots()
            for o in args.ords_use:
                ax.plot(wraw[o], fraw[o], 'C0')
                ax.plot(w[o], fconv[o], 'C1')
            plt.tight_layout()
            plt.show()
            plt.close()

    # -----------------------------------------------------------------------------

    # Normalize spectrum template

    # Regions to ignore to find continuum
    # TO DO: Load from file option
    # Add NIR lines

    fmask = [i.copy() for i in f]
    lines_mask_star, lines_mask_tell, lines_mask_tell_file, lines_mask_pixblue, lines_mask = [], [], [], [], []
    if args.cont_linesmaskstar or args.cont_linesmasktell or args.cont_pixblue:

        # Stellar lines to ignore. Depend on star RV: tplrv
        if args.cont_linesmaskstar:
            lines_mask_star = [
                # Line central wavelength, region to block at each side in km/s, tag
                (5877.25, 100., 'HeID3'),
                (5891.583264, 100., 'NaID1'),
                (5897.558147, 100., 'NaID2'),
                (6564.61,     200., 'Halpha'),
                # Order 32: Selected by eye regions with low flux
                (7056.75, 30, ''),
                (7090.08, 30, ''),
                (7127.95, 30, ''),
            ]
            # Shift stellar lines
            lines_mask_star_new = []
            for i in lines_mask_star:
                lines_mask_star_new.append((i[0]*(1.+tplrv/C_KMS), i[1], i[2]))
            lines_mask_star = lines_mask_star_new

        # Telluric regions to ignore. Do not need to move.
        if args.cont_linesmasktell:
            lines_mask_tell = [
                # Line central wavelength, region to block at each side in km/s, tag
                # Order 38: Selected by eye regions with low flux
                (7606.75, 470, ''),  # Absorption band: 7594.78 -- 7618.72
                # (7639.22, 300, ''),  # Absorption band: 7621.06 -- 7657.38
            ]
        # Other fixed regions to ignore from file
        if args.cont_linesmasktell_read is not None:
            lines_mask_tell_file = list(np.genfromtxt(args.cont_linesmasktell_read, delimiter=',', dtype=[float, float, str]))

        # Blue pixels to ignore
        if args.cont_pixblue:
            # Check if args_cont_pixblue_ords in ords
            # ...
            #
            lines_mask_pixblue = []
            for o in args.cont_pixblue_ords:
                # Find wavelength closest to pixel
                dw_i = (w[o][args.cont_pixblue_npix] - w[o][0]) / 2.
                w_i = np.nanmean(w[o][0:args.cont_pixblue_npix+1])
                dv_i = C_KMS * dw_i / w_i
                lines_mask_pixblue.append((w_i, dv_i, 'pixblue{}o{}'.format(args.cont_pixblue_npix, o)))

        # Join lines
        lines_mask = lines_mask_star + lines_mask_tell + lines_mask_tell_file + lines_mask_pixblue

        lines_mask = pd.DataFrame(lines_mask, columns=['wcen', 'dv', 'name'])
        # Compute w limits of regions to block around lines
        lines_mask['w1'] = lines_mask['wcen']*(1.-lines_mask['dv']/C_KMS)
        lines_mask['w2'] = lines_mask['wcen']*(1.+lines_mask['dv']/C_KMS)

        # Change flux to nan in regions specified in `lines_mask`
        for i in lines_mask.index:
            for o in args.ords_use:
                if lines_mask.loc[i]['wcen'] >= w[o][0] and lines_mask.loc[i]['wcen'] <= w[o][-1]:  # If line in this order
                    i1 = np.argmin(np.abs(w[o]-lines_mask.loc[i]['w1']))
                    i2 = np.argmin(np.abs(w[o]-lines_mask.loc[i]['w2']))
                    # mask = np.ones_like(f[o], dtype=bool)
                    for j in np.arange(i1, i2+1, 1):
                        fmask[o][j] = np.nan

    # Normalize
    if args.cont is not None:
        verboseprint('\nContinuum normalize spectrum.')
        verboseprint('Some regions removed when finding the continuum')
        # Use `fmask` instead of `f` to not consider bad regions when normalizing
        fcontinuumnorm, continuum, Continuum, f_medfilt, f_maxfilt, idxmax, fitpar = spectrumutils.fitcontinuum_echelle(w, fmask, medfiltsize=args.contfiltmed, maxfiltsize=args.contfiltmax, fitfunc=args.cont, polyord=args.contpolyord, splsmooth=args.contsplsmooth, spldegree=args.contspldegree, ords_use=args.ords_use, returnfill=True)
        # Normalize actual flux array `f` (not `fmask`!)
        f = [f[o]/continuum[o] for o in ords]
        # f = [fcontinuumnorm[o].copy() for o in ords]

    # Plot norm spec
    # if True:
    if args.plot_test:
        verboseprint('Normalised template')
        fig, ax = plt.subplots(2, 1, sharex=True)
        xmin, xmax = np.nanmin([np.nanmin(w[o]) for o in args.ords_use]), np.nanmax([np.nanmax(w[o]) for o in args.ords_use])
        ymin0, ymax0 = np.nanmin([np.nanmin(fraw[o]) for o in args.ords_use]), np.nanmax([np.nanmax(fraw[o]) for o in args.ords_use])
        ymin1, ymax1 = np.nanmin([np.nanmin(f[o]) for o in args.ords_use]), np.nanmax([np.nanmax(f[o]) for o in args.ords_use])
        for o in args.ords_use:
            # Spectrum raw
            ax[0].plot(wraw[o], fraw[o])
            ax[0].plot(w[o], continuum[o])
            # Spectrum continuum norm
            ax[1].plot(w[o], f[o])
        # Telluric mask
        if args.filtell is not None:
            telluricutils.plot_mask(wt, ft*ymax0, ax=ax[0], leglab='Telluric mask', alpha=.3, color='k')
            telluricutils.plot_mask(wt_broaden_join, ft_broaden_join*ymax0, ax=ax[0], leglab='Telluric mask broaden {:.0f}'.format(tellbroadendv), alpha=.3, color='k')
            telluricutils.plot_mask(wt, ft*ymax1, ax=ax[1], leglab='Telluric mask', alpha=.3, color='k')
            telluricutils.plot_mask(wt_broaden_join, ft_broaden_join*ymax1, ax=ax[1], leglab='Telluric mask broaden {:.0f}'.format(tellbroadendv), alpha=.3, color='k')

        if len(lines_mask) > 0:
            for i in lines_mask.index:
                ax[1].vlines(lines_mask.loc[i]['w1'], 0, 1, colors='b', zorder=10)
                ax[1].vlines(lines_mask.loc[i]['w2'], 0, 1, colors='r', zorder=10)

        for a in ax:
            a.set_xlim(xmin, xmax)
        ax[0].set_ylim(ymin0, ymax0)
        ax[1].set_ylim(ymin1, ymax1)
        plt.tight_layout()
        plt.show()
        plt.close()

    # Save processed template
    filout = os.path.join(args.dirout_tpl, os.path.splitext(os.path.basename(args.filtpl))[0] + 'matrix.pkl')
    spectrumutils.spec_save_pkl_matrix(w, f, filout, verb=args.verbose)

    ###############################################################################

    # Find spectrum minima and maxima
    verboseprint('\nFind template minima and maxima')

    imin, imax1, imax2 = peakutils.find_abspeaks_echelle(f, method='custom', ords_use=args.ords_use)

    ###############################################################################

    # fig, ax = plt.subplots()
    # for o in args.ords_use:
    #     ax.plot(w[o], f[o], 'k')
    #     ax.vlines(w[o][imin[o]], 0, 1, colors='C1')
    #     ax.vlines(w[o][imax1[o]], 0, 1, colors='b', linestyles='dashed')
    #     ax.vlines(w[o][imax2[o]], 0, 1, colors='r', linestyles='dashed')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    # sys.exit()

    # Fit minima

    if args.filfit is None:
        verboseprint('\nFit all minima')

        #for o in args.ords_use:
        #    out = spectrumlib.fit_gaussian_spec(w, f, imin, imax1, imax2, nfitmin=4, amp_hint=-0.5, cen_hint=None, wid_hint=0.01, shift_hint=0.8, minmax='min', returntype='pandas')

        datafit = [[]]*nord
        for o in ords:
            if o in args.ords_use and len(imin[o] > 0):

                # Fit
                datafit_o = peakutils.fit_gaussian_spec(w[o], f[o], imin[o], imax1[o], imax2[o], nfitmin=4, amp_hint=-0.2, cen_hint=w[o][imin[o]], wid_hint=0.4, shift_hint=1., minmax='min', returntype='pandas', barmsg=' Fit min o{:02d}'.format(o))

                # Convert FWHM from wavelength [A] to velocity [m/s]
                #  FWHM_wav = dw; dw/w = dv/c => FWHM_vel = dv = c*dw/w
                #  Use postion of line minimum from fit (`cen`) as w
                datafit_o['fwhm_kms'] = C_KMS * datafit_o['fwhm'] / datafit_o['cen']

                # Flag nan and inf: add new column 'flagnan', True if contains nan, False otherwise
                # - Replace inf with nan
                datafit_o = datafit_o.replace([np.inf, -np.inf], np.nan)
                # - Flag nan: True contains nan, False otherwise
                datafit_o['flagnan'] = datafit_o[['amp', 'amperr', 'cen', 'cenerr', 'shift', 'shifterr', 'wid', 'widerr']].isnull().any(axis=1)

                # Flag lines (imin, imax1 or imax2) in bad (telluric, emission) regions: True in bad region, False otherwise
                maskmin = np.array(Maskbad_inv(datafit_o['wmin']), dtype=bool)
                maskmax1 = np.array(Maskbad_inv(datafit_o['wmax1']), dtype=bool)
                maskmax2 = np.array(Maskbad_inv(datafit_o['wmax2']), dtype=bool)
                datafit_o['flagbad'] = ~(maskmin & maskmax1 & maskmax2)

                # Save
                fil = os.path.join(args.dirout_minfit, 'minfito{:02d}.dat'.format(o))
                datafit_o.to_csv(fil, sep=' ', na_rep=np.nan, float_format=None, columns=None, header=True)

                datafit[o] = datafit_o

            else:
                # Save empty file
                fil = os.path.join(args.dirout_minfit, 'minfito{:02d}.dat'.format(o))
                os.system('touch {}'.format(fil))

    else:
        verboseprint('\nRead minima fit from {}'.format(args.filfit))

        datafit = [[]]*nord
        for o in ords:
            # Check if file exists
            if not os.path.isfile(os.path.join(args.filfit, 'minfito{:02d}.dat'.format(o))):
                sys.exit('Minima fit file `{}` does not exist.'.format(args.filfit, 'minfito{:02d}.dat'.format(o)))

            # Read
            try: datafit[o] = pd.read_csv(os.path.join(args.filfit, 'minfito{:02d}.dat'.format(o)), sep='\s+', header=0, names=None, index_col=0, usecols=None)
            # if file empty
            except: datafit[o] = []

            # Check that the minimum indices of the file are the same as the ones in the spectrum
            if o in args.ords_use and len(imin[o]) > 0:
                if not np.array_equiv(datafit[o].index, imin[o]): 
                    sys.exit('Order {} different minima in file {} than in template {}'.format(o, os.path.join(args.filfit, 'minfito{:02d}.dat'.format(o)), args.filtpl))

    # Plot tpl with all and selected minima
    # Plot tpl with fits
    if args.plot_test:
        if 12 in args.ords_use:
            o = 12
        elif 26 in args.ords_use:
            o = 26
        else:
            o = args.ords_use[0]
        fig, ax = plt.subplots()
        ymin, ymax = min(f[o]), max(f[o])
        xmin, xmax = min(w[o]), max(w[o])
        # Spectrum
        ax.plot(w[o], f[o], label='Spec')
        # # All minima
        # ax.vlines(datafit[o]['wmin'], 0, datafit[o]['fmin'], colors='C1', linestyles='dashed', alpha=.3)
        # ax.vlines(datafit[o]['wmax1'], 0, datafit[o]['fmax1'], colors='C2', linestyles='dashed', alpha=.3)
        # ax.vlines(datafit[o]['wmax2'], 0, datafit[o]['fmax2'], colors='C2', linestyles='dashed', alpha=.3)
        # Clean minima
        ax.vlines(datafit[o]['wmin'], 0, datafit[o]['fmin'], colors='C1')
        ax.vlines(datafit[o]['wmax1'], 0, datafit[o]['fmax1'], colors='C2')
        ax.vlines(datafit[o]['wmax2'], 0, datafit[o]['fmax2'], colors='C2')

        # COMMENT
        # Depth
        # ax.vlines(datafit[o]['cen'], 0, datafit[o]['depth'], color='C3', linestyles='dashed')

        # Fits
        for m in range(len(datafit[o])):
            x = np.arange(datafit[o]['wmax1'].iloc[m], datafit[o]['wmax2'].iloc[m]+0.01, 0.01)
            ax.plot(x, peakutils.gaussian(x, amp=datafit[o]['amp'].iloc[m], cen=datafit[o]['cen'].iloc[m], wid=datafit[o]['wid'].iloc[m], shift=datafit[o]['shift'].iloc[m]), 'r-')

        # Telluric mask
        if args.filtell is not None:
            telluricutils.plot_mask(wt_broaden_join, ft_broaden_join*ymax, ax=ax, leglab='Telluric mask', alpha=.3, color='k')
        ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.show()
        plt.close()

    # -----------------------------------------------------------------------------

    # Select minima without nan and not in bad regions
    # Add flag 'flag': True if wrong datapoint, False otherwise
    verboseprint('\nFlag "bad" minima: nan, inf, telluric, other selected regions')

    for o in ords:
        if o in args.ords_use and len(imin[o]) > 0:
            datafit[o]['flag'] = ~(~datafit[o]['flagnan'] & ~datafit[o]['flagbad'])

            # if args.filtell is not None:
            #     datafit[o]['flag'] = ~((~datafit[o]['flagnan']) & (~datafit[o]['flagtell{:.0f}'.format(tellbroadendv)]))
            # else:
            #     datafit[o]['flag'] = datafit[o]['flagnan']
        else:
            pass

    ###############################################################################

    # Minima parameters
    verboseprint('\nCompute extra minima parameters')

    # From the Gaussian fit: 'amp', 'cen', 'shift', 'wid', 'fwhm'
    #
    # Peak properties definition: 1 is at the continuum, 0 is the minimum of the peak.
    #
    # 1 _|_    _____           __
    #    | \  /     \    ___  /
    #    |  \/       \  /   \/
    # 0 _|            \/
    #
    # - `fmin`: Flux value of the minimum of the line.
    # - `depth`: Flux value of the minimum of the Gaussian fit to the line. Should be similar to `fmin`; if it differs, it means that the fit was not good.
    # - `contrast`: Vertical distance of the line minimum to the neighbouring maxima (so actually there are 2 values, one for each side: `contrast1`, `contrast2`)

    # - height: Vertical distance of the peak to the continuum (1).

    # ---- ToDo Scipy peaks ----
    # - threshold: Vertical distance of the peak to its neighbouring points.
    # - distance: Horizontal distance between neighbouring peaks.
    # - prominence: The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line. Equals `contrastmax`.
    #
    # - depth: Continuum 1, minima peak -> 0

    lisminparam = [
        'wmin', 'fmin',
        'amp', 'cen', 'shift', 'wid', 'fwhm', 'fwhm_kms', 'depth',
        'contrast1', 'contrast2', 'contrastmin', 'contrastmax', 'contrastmean',
        'prominence', 'left_base', 'right_base', 'contour_heights',
        'ew',
        'contrastmeanfwhm-1',
        # 'height',
        # 'threshold',
    ]

    # for o in args.ords_use:
    #     fig, ax = plt.subplots()
    #     ax.hist(datafit[o]['depth'])
    #     plt.show()
    #     plt.close()
    # #datafit[o]['shift']

    for o in ords:
        if o in args.ords_use and len(imin[o]) > 1:
            # More parameters

            datafit[o]['depth'] = datafit[o]['amp']+datafit[o]['shift']

            datafit[o]['contrastfit'] = -(datafit[o]['amp']/datafit[o]['shift'])
            datafit[o]['contrast1'] = datafit[o]['fmax1']-datafit[o]['fmin']
            datafit[o]['contrast2'] = datafit[o]['fmax2']-datafit[o]['fmin']
            datafit[o]['contrastmin'] = datafit[o][['contrast1', 'contrast2']].min(axis=1)
            datafit[o]['contrastmax'] = datafit[o][['contrast1', 'contrast2']].max(axis=1)
            datafit[o]['contrastmean'] = datafit[o][['contrast1', 'contrast2']].mean(axis=1)  # Same as (datafit[o]['fmax1']+datafit[o]['fmax2'])/2.-datafit[o]['fmin']

            datafit[o]['prominence'], datafit[o]['left_base'], datafit[o]['right_base'] = scipy.signal.peak_prominences(-f[o], datafit[o].index)
            datafit[o]['contour_heights'] = f[o][datafit[o].index] + datafit[o]['prominence']  # i.e. -(-f - prominence)

            datafit[o]['ew'] = ((datafit[o]['fmax1']+datafit[o]['fmax1'])/2.-datafit[o]['fmin'])*datafit[o]['fwhm_kms']
            datafit[o]['contrastmeanfwhm-1'] = datafit[o]['contrastmean']/datafit[o]['fwhm_kms']

            # TO DO
            # Add more: line separation...
            #datafit[o]['threshold'] = datafit[o]['']

    ###############################################################################

    # Remove "bad" lines: minima with unphysical values, e.g. negative widths or negative depths, and lines marked with `flag` (see above)
    verboseprint('\nRemove minima with unphysical values (bad fits)')

    lineclean_fwhmmin, lineclean_fwhmmax = 0., 200.
    lineclean_depthmin, lineclean_depthmax = 0., 1.0
    lineclean_contrastmin, lineclean_contrastmax = 0., 1.

    datafit_notclean = [datafit[o] for o in ords]  # Keep all data
    # datafit = [datafit_notclean[o] for o in ords]  # Keep all data
    condclean = [[]]*nord
    for o in ords:
        if o in args.ords_use and len(imin[o]) > 1:
            condclean_fwhm = (datafit[o]['fwhm_kms'] >= lineclean_fwhmmin) & (datafit[o]['fwhm_kms']<=lineclean_fwhmmax)
            condclean_depth = (datafit[o]['depth'] >= lineclean_depthmin) & (datafit[o]['depth']<=lineclean_depthmax)
            condclean_contrast1 = (datafit[o]['contrast1'] >= lineclean_contrastmin) & (datafit[o]['contrast1'] <= lineclean_contrastmax)
            condclean_contrast2 = (datafit[o]['contrast2'] >= lineclean_contrastmin) & (datafit[o]['contrast2'] <= lineclean_contrastmax)

            # Final selection
            condclean[o] = condclean_fwhm & condclean_depth & condclean_contrast1 & condclean_contrast2 & ~datafit[o]['flag']

            # Clean
            datafit[o] = datafit[o][condclean[o]]

    ###############################################################################

    # Save clean minima fit data

    for o in ords:
        if o in args.ords_use and len(imin[o] > 0):

            # Save
            fil = os.path.join(args.dirout_minfit, 'minfito{:02d}_clean.dat'.format(o))
            datafit[o].to_csv(fil, sep=' ', na_rep=np.nan, float_format=None, columns=None, header=True)

        else:
            # Save empty file
            fil = os.path.join(args.dirout_minfit, 'minfito{:02d}_clean.dat'.format(o))
            os.system('touch {}'.format(fil))

    ###############################################################################

    # Select minima
    verboseprint('\nSelect minima based on some conditions')

    cond = [[]]*nord
    for o in args.ords_use:

        # Contrast
        # --------
        # Cut same values for all orders
        condcontrast1 = (datafit[o]['contrast1'] >= args.line_contrast1min) & (datafit[o]['contrast1'] <= args.line_contrast1max)
        condcontrast2 = (datafit[o]['contrast2'] >= args.line_contrast2min) & (datafit[o]['contrast2'] <= args.line_contrast2max)
        condcontrastmin = (datafit[o]['contrastmin'] >= args.line_contrastminmin) & (datafit[o]['contrastmin'] <= args.line_contrastminmax)
        condcontrastmax = (datafit[o]['contrastmax'] >= args.line_contrastmaxmin) & (datafit[o]['contrastmax'] <= args.line_contrastmaxmax)
        condcontrastmean = (datafit[o]['contrastmean'] >= args.line_contrastmeanmin) & (datafit[o]['contrastmean'] <= args.line_contrastmeanmax)

        # FWHM
        # ----
        # Cut same values for all orders
        condfwhm = (datafit[o]['fwhm_kms'] >= args.line_fwhmmin) & (datafit[o]['fwhm_kms'] <= args.line_fwhmmax)

        # Depth
        # -----
        # Lines with physical depth
        conddepth = (datafit[o]['depth'] >= args.line_depthmin) & (datafit[o]['depth'] <= args.line_depthmax)

        # Cut depth values depending on the wavelength

        # Define new 0 depth for this order

        # Select minima with smaller depth (i.e. deepest minima).
        # The number of minima to select, in percentage, is args.line_depthw_percentdeepest. E.g. args.line_depthw_percentdeepest=0.1 means 10 % of the deepest minima
        nmindeep = int(len(datafit[o][conddepth]['depth'])*args.line_depthw_percentdeepest)  # Number of minima
        mindeep = np.sort(datafit[o][conddepth]['depth'])[:nmindeep]  # The depths of the minima
        # Select which method use to select the new depth0
        if args.line_depthw_type == 'constant':
            # Compute the mean of these deepest minima
            # -> New 0 depth
            # -> New depth range: from 1 to depthnew0, while before it was from 1 to 0
            depthnew0 = np.mean(mindeep)

            # Select lines with depth smaller than the args.line_depthw_depthmaxquantile quantile of the new depth range (not of the values of the depth)
            # New depth limit
            depthmaxnew = np.quantile([depthnew0, 1.], args.line_depthw_depthmaxquantile)
            # depthmaxnew = 1-(1.-depthnew0)*(1-args.line_depthw_depthmaxquantile)
            depthcondw = pd.Series(np.zeros(len(datafit[o].index), dtype=bool), index=datafit[o].index)  # Bool array all elements False
            for i in datafit[o].index:
                if datafit[o].loc[i, 'depth'] <= depthmaxnew:
                    depthcondw[i] = True
        elif args.line_depthw_type == 'poly':
            # Fit a poly of order args.line_depthw_polyord
            # The poly will be the new depth0, changing with w within the order
            sys.exit('Not implemented yet')

        # -----------------------------------------------------

        # Final selection
        if args.condnodepth is False:  # Take all conditions into account
            cond[o] = condcontrast1 & condcontrast2 & condcontrastmin & condcontrastmax & condcontrastmean & condfwhm & conddepth & depthcondw
        else:  # Do not take depth conditions into account (in cases with non-normalised templates)
            cond[o] = condcontrast1 & condcontrast2 & condcontrastmin & condcontrastmax & condcontrastmean & condfwhm

    # Final lines (cond = True)
    datafit_final = [[]]*nord
    for o in args.ords_use:
        datafit_final[o] = datafit[o][cond[o]]

    # Plot depth selection
    if args.plot_test:
        fig, ax = plt.subplots()
        for o in args.ords_use:
            # Spectrum
            ax.plot(w[o], f[o], 'k', alpha=0.8 if o % 2 == 0 else 1)

            # All minima
            ax.vlines(datafit[o]['cen'], 0, datafit[o]['depth'], colors='b', linestyles='dashed', alpha=.4, label='', zorder=0)

            # Compute new depth 0
            conddepth = (datafit[o]['depth'] >= args.line_depthmin) & (datafit[o]['depth'] <= args.line_depthmax)
            nmindeep = int(len(datafit[o][conddepth]['depth'])*args.line_depthw_percentdeepest)  # Number of minima
            mindeep = np.sort(datafit[o][conddepth]['depth'])[:nmindeep] # The depths of the minima
            depthnew0 = np.mean(mindeep) # Compute the mean of these deepest minima
            # New depth0
            ax.hlines(depthnew0, min(w[o]), max(w[o]), colors='C1', zorder=10, label='')
            # Depth maximum
            ax.hlines(np.quantile([depthnew0, 1.], args.line_depthw_depthmaxquantile), min(w[o]), max(w[o]), colors='C2', zorder=10, label='')

            # Selected minima
            ax.vlines(datafit[o]['cen'][cond[o]], 0, datafit[o]['depth'][cond[o]], colors='C3', label='', zorder=0)

        # Tellurics
        if args.filtell is not None:
            telluricutils.plot_mask(wt_broaden_join, ft_broaden_join*1, ax=ax, leglab='Telluric mask broaden {:.0f}'.format(tellbroadendv), alpha=.3, color='k')
        # ax.vlines(wmold, 0, fmold, colors='C3', linestyles='dashed', lw=3, label='old mask', zorder=0)
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    # Minima characterization
    if args.plot_test:

        sh_leg = False
        if sh_leg == False: legstr = ''
        else: legstr = '_leg'

        # Select minima
        if args.inst == 'CARM_VIS':
            liswminapprox = [
                6788.55,
                6625.68, 6976.54, 7500.97, 7532.31,  # J07274+052
                ]  # [A]
        else:
            liswminapprox = []

        current_lw = mpl.rcParams['lines.linewidth']

        for wminapprox in liswminapprox:

            fig, ax = plt.subplots(figsize=(8, 6))

            # Get order
            ouse = None
            for o in args.ords_use:
                if wminapprox >= w[o][0] and wminapprox <= w[o][-1]:
                    ouse = o
                    break
            if ouse is None: continue
            # Find indices
            idx = np.abs(datafit[ouse]['cen'] - wminapprox).idxmin()

            # Minima limits
            # - x
            w1 = datafit[ouse]['wmax1'].loc[idx]
            w2 = datafit[ouse]['wmax2'].loc[idx]
            dw = w2 - w1
            padx = 0.5  # [%]
            xmin = w1 - dw*padx
            xmax = w2 + dw*padx
            # - y
            f1 = datafit[ouse]['fmin'].loc[idx]
            f2 = np.nanmax([datafit[ouse]['fmax1'].loc[idx], datafit[ouse]['fmax2'].loc[idx]])
            df = f2 - f1
            pady = 0.3  # [%]
            ymin = f1 - df*pady
            ymax = f2 + df*pady

            mpl.rcParams['lines.linewidth'] = 3

            # Tpl
            # ax.step(w[ouse], f[ouse], 'k', where='mid', color=color, label='Template', zorder=10)
            ax.plot(w[ouse], f[ouse], 'k', marker='.', ms=18, color='k', label='Template', zorder=10, lw=mpl.rcParams['lines.linewidth']+1)

            # Minima, maxima
            ax.vlines(datafit[ouse]['wmin'].loc[idx], 0, 1, linestyles='dashdot', colors='0.4')  # label='Minimum, maxima')
            ax.vlines(datafit[ouse]['wmax1'].loc[idx], 0, 1, linestyles='dotted', colors='0.4')
            ax.vlines(datafit[ouse]['wmax2'].loc[idx], 0, 1, linestyles='dotted', colors='0.4')

            # Shaded region from max1 to max2
            # wmax, fmax = telluricutils.wlimits2mask(datafit[ouse]['wmax1'].values, datafit[ouse]['wmax2'].values, dw=0.001)
            # ax.fill_between(wmax, fmax, zorder=0, color='0.7')
            ax.axvspan(w1, w2, color='0.9', zorder=0)

            # Fit
            fitx = np.arange(w1, w2+0.01, 0.01)
            fity = peakutils.gaussian(fitx, amp=datafit[ouse]['amp'].loc[idx], cen=datafit[ouse]['cen'].loc[idx], wid=datafit[ouse]['wid'].loc[idx], shift=datafit[ouse]['shift'].loc[idx])
            ax.plot(fitx, fity, color='r', label='Fit', zorder=11, alpha=.8)

            # Fit params: cen
            ax.vlines(datafit[ouse]['cen'].loc[idx], 0, 1, linestyles='dashed', colors='r')  #, label='Fit minimum')

            # Fit params: FWHM

            # Depth

            # Contrast

            # ...

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.ticklabel_format(useOffset=False)
            if sh_leg: ax.legend(loc='upper left').set_zorder(1000)
            ax.set_xlabel('Wavelength {}'.format(angstromstrunit))
            ax.set_ylabel('Norm. flux')
            plotutils.figout(fig, filout=os.path.join(args.dirout_plot, 'line_characterization_w{:05.0f}-{:05.0f}{}'.format(xmin, xmax, legstr)), sv=args.plot_sv, svext=args.plot_ext, sh=True)

        mpl.rcParams['lines.linewidth'] = current_lw

    # Plot lines
    if args.plot_finalslice:

        fig, ax = plt.subplots(figsize=(16, 6))

        axmask = ax.twinx()
        # # Set ax's patch invisible
        # ax.patch.set_visible(False)
        # # move ax in front
        # ax.set_zorder(axmask.get_zorder() + 1)

        xmin, xmax = np.nanmin([np.nanmin(w[o]) for o in args.ords_use]), np.nanmax([np.nanmax(w[o]) for o in args.ords_use])
        ymin, ymax = np.nanmin([np.nanmin(f[o]) for o in args.ords_use]), np.nanmax([np.nanmax(f[o]) for o in args.ords_use])
        yminmask, ymaxmask = np.nanmin([np.nanmin(datafit[o][cond[o]]['contrastmeanfwhm-1']) for o in args.ords_use if len(datafit[o][cond[o]]['contrastmeanfwhm-1']) > 0]), np.nanmax([np.nanmax(datafit[o][cond[o]]['contrastmeanfwhm-1']) for o in args.ords_use if len(datafit[o][cond[o]]['contrastmeanfwhm-1']) > 0])
        # for o in [5,6]:
        # for o in [48]:
        for o in args.ords_use:
            # Spectrum
            if o % 2 == 0: color = 'k'
            else: color = 'k'  #'0.4'
            if o == args.ords_use[0]:
                labspec = 'Template'
                lablin = 'Minima'
                labmask = 'Mask'
            else:
                labspec = ''
                lablin = ''
                labmask = ''
            # ax.step(w[o], f[o], 'k', where='mid', color=color, label=labspec, zorder=10)
            ax.plot(w[o], f[o], 'k', marker='.', ms=8, color=color, label=labspec, lw=2, zorder=10)

            # Clean minima
            ax.vlines(datafit[o]['cen'], 0, datafit[o]['depth'], colors='C2', linestyles='dashed', label=lablin, lw=3, zorder=1)
            # Selected minima
            # Plot mask flux scale in another yaxis
            colormask = 'C1'
            axmask.vlines(datafit[o][cond[o]]['cen'], 0, datafit[o][cond[o]]['contrastmeanfwhm-1'], colors=colormask, label=labmask, lw=4, zorder=2, alpha=.9)
            # ax.vlines(datafit[o][cond[o]]['cen'], 0, datafit[o][cond[o]]['depth'], colors='C1', label=labmask, zorder=2)

        # Telluric mask
        if args.filtell is not None:
            # labtell = 'Telluric mask'
            labtell = ''
            telluricutils.plot_mask(wt_broaden_join, ft_broaden_join*ymax, ax=ax, leglab=labtell, color='0.7', zorder=0)

        ax.minorticks_on()
        axmask.minorticks_on()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        axmask.set_ylim(0, ymaxmask)

        # ax.legend().set_zorder(200)
        fig.legend(loc=1, bbox_to_anchor=(0.5, 1), bbox_transform=ax.transAxes, ncol=3)
        ax.set_xlabel('Wavelength {}'.format(angstromstrunit))
        ax.set_ylabel('Norm. flux')
        axmask.set_ylabel('Mask weight', color=colormask)

        axmask.tick_params(axis='y', which='both', color=colormask, labelcolor=colormask)
        # axmask.spines['right'].set_color(colormask)

        # Plot some representative regions
        # (6175.16, 6184.57)
        # (7713.28, 7725.69)
        # (7720.92, 7733.41)
        # if args.obj == 'J12123+544S':
        #     region = [
        #         [(7720.92, 7733.41), (0.6, 1.08)],
        #     ]
        # elif args.obj == 'J07274+052':
        #     region = [
        #         [(7720.92, 7733.41), (0.4, 1.08)],
        #     ]
        region = [
            # [(x0, x1), (y0, y1)],
            [(xmin, xmax), (0., 10.)],
            [(7540, 7575), (0., 10.)],
            [(7720.92, 7731), (0.4, 1.08)],
            # [(7720.92, 7733.41), (0.4, 1.08)],
            ]

        for dx, dy in region:
            ax.set_xlim(dx)
            ax.set_ylim(dy)
            plotutils.figout(fig, filout=os.path.join(args.dirout_plot, 'tpl_allmin_mask_w{:05.0f}-{:05.0f}'.format(dx[0], dx[1])), sv=args.plot_sv, svext=args.plot_ext, sh=args.plot_sh)

    # Join orders
    verboseprint('\nJoin orders')

    # only data with more than 1 order
    if len(datafit_final) > 1:
        if args.joinord == 'merge':

            # Join merging blue and red extremes of consecutive orders
            wm_join, fm_join = [], []
            lineseparationlimit = args.dwmin

            for i in range(len(args.ords_use)):

                # Current order
                o = args.ords_use[i]
                # Previous order
                if o == args.ords_use[0]: oprev = np.nan
                else: oprev = args.ords_use[i-1]
                # Next order
                if o == args.ords_use[-1]: onext = np.nan
                else: onext = onext = args.ords_use[i+1]

                # print('-- o', o)
                # First order
                if o == args.ords_use[0]:

                    # No blue lines in overlap with previous order, because this is the first one
                    # Red lines in overlap with next order (include now!)
                    overlapred = datafit_final[o]['cen'] >= datafit_final[onext]['cen'].min()
                    # No overlap lines
                    datafito_nooverlap = datafit_final[o][~overlapred]

                    # print('-- -- total:', len(datafit_final[o]))
                    # print('-- -- in overlap:', len(datafit_final[o][overlapred]))
                    # print('-- -- no overlap:', len(datafit_final[o][~overlapred]))

                    # Add lines not in overlap region
                    wm_join.extend(datafito_nooverlap[args.maskwave])
                    fm_join.extend(datafito_nooverlap[args.maskweight])
                    # print('-- -- add no overlap lines:', len(datafito_nooverlap[args.maskwave]))

                    # Check if there is overlap with next order
                    # - If no overlap, continue
                    # - If there is overlap, go line by line, see if line exists in next order, and merge them
                    if True in overlapred.values:
                        # Next order minima wavelength in the overlap region
                        nextord_overlap = datafit_final[onext]['cen'][datafit_final[onext]['cen'] <= datafit_final[o]['cen'].max()]
                        # print('-- -- -- overlap, possible common line in next order:', len(nextord_overlap))
                        s = 0
                        for l in datafit_final[o][overlapred].index:
                            # Find closest line
                            idxmindist = (np.abs(datafit_final[o].at[l, 'cen'] - nextord_overlap)).idxmin()  # idx closest line
                            mindist = np.min(np.abs(datafit_final[o].at[l, 'cen'] - nextord_overlap))  # distance
                            if mindist <= lineseparationlimit:  # If lines 2 orders close enough, slelect line and merge values
                                wm_join.append(np.nanmean([datafit_final[o].at[l, args.maskwave], datafit_final[onext].at[idxmindist, args.maskwave]]))
                                fm_join.append(np.nanmean([datafit_final[o].at[l, args.maskweight], datafit_final[onext].at[idxmindist, args.maskweight]]))
                                s += 1
                        # print('-- -- -- lines in overlap added:', s)

                # Central orders
                elif o != args.ords_use[0] and o != args.ords_use[-1]:
                    # Blue lines in overlap with previous order, already dealt with (not include now!)
                    overlapblue = datafit_final[o]['cen'] <= datafit_final[oprev]['cen'].max()
                    # Red lines in overlap with next order (include now!)
                    overlapred = datafit_final[o]['cen'] >= datafit_final[onext]['cen'].min()
                    # No overlap lines
                    datafito_nooverlap = datafit_final[o][(~overlapblue) & (~overlapred)]

                    # print('-- -- total:', len(datafit_final[o]))
                    # print('-- -- in overlap:', len(datafit_final[o][overlapblue]), len(datafit_final[o][overlapred]))
                    # print('-- -- no overlap:', len(datafit_final[o][(~overlapblue) & (~overlapred)]))

                    # Add lines not in overlap region
                    wm_join.extend(datafito_nooverlap[args.maskwave])
                    fm_join.extend(datafito_nooverlap[args.maskweight])
                    # print('-- -- add no overlap lines:', len(datafito_nooverlap[args.maskwave]))

                    # Check if there is overlap with next order
                    # - If no overlap, add all order (except the bluest lines which could have overlap with the previous order)
                    # - If there is overlap, go line by line, see if line exists in next order, and merge them
                    if True in overlapred.values:
                        # overlap = datafit[o][cond[o]]['cen'] >= datafit[onext][cond[onext]]['cen'].min()
                        # Next order minima wavelength in the overlap region
                        nextord_overlap = datafit_final[onext]['cen'][datafit_final[onext]['cen'] <= datafit_final[o]['cen'].max()]
                        # print('-- -- -- overlap, possible common line in next order:', len(nextord_overlap))
                        # if len(nextord_overlap) == 0:  # If no minima in overlap region, skip to next order. Should not happen here, checked before
                        #     continue
                        # for l in datafit[o][cond[o]][overlap].index:
                        s = 0
                        for l in datafit_final[o][overlapred].index:
                            # Find closest line
                            # idxmindist = np.argmin(np.abs(datafit[o].at[l, 'cen'] - nextord_overlap))  # idx closest line
                            idxmindist = (np.abs(datafit_final[o].at[l, 'cen'] - nextord_overlap)).idxmin()  # idx closest line
                            mindist = np.min(np.abs(datafit_final[o].at[l, 'cen'] - nextord_overlap))  # distance
                            if mindist <= lineseparationlimit:  # If lines 2 orders close enough, slelect line and merge values
                                wm_join.append(np.nanmean([datafit_final[o].at[l, args.maskwave], datafit_final[onext].at[idxmindist, args.maskwave]]))
                                fm_join.append(np.nanmean([datafit_final[o].at[l, args.maskweight], datafit_final[onext].at[idxmindist, args.maskweight]]))
                                s += 1
                        # print('-- -- -- lines in overlap added:', s)

                # Last order: No overlap with any other order (overlap blue part already dealt with)
                elif o == args.ords_use[-1]:

                    # Blue lines in overlap with previous order, already dealt with (not include now!)
                    overlapblue = datafit_final[o]['cen'] <= datafit_final[oprev]['cen'].max()

                    # print('-- -- total:', len(datafit_final[o]))
                    # print('-- -- in overlap:', len(datafit_final[o][overlapblue]))
                    # print('-- -- no overlap:', len(datafit_final[o][~overlapblue]))

                    # No red lines in overlap with next order, because this is the last one
                    # No overlap lines
                    datafito_nooverlap = datafit_final[o][~overlapblue]
                    wm_join.extend(datafito_nooverlap[args.maskwave])
                    fm_join.extend(datafito_nooverlap[args.maskweight])
                    # print('-- -- add no overlap lines:', len(datafito_nooverlap[args.maskwave]))

        elif args.joinord == 'keepred':
            sys.exit('Not implemented yet!')
            # TODO REVISE
            # # Join keeping the redder order lines in the overlap region (bc bluer part of the orders more noisy)
            # wm_join, fm_join = [], []
            # for o in args.ords_use:
            #     if o==args.ords_use[0]:  #1st ord keep all range
            #         for i in datafit[o].index[cond[o]]:
            #             wm_join.append(datafit[o].at[i, args.maskwave])
            #             fm_join.append(datafit[o].at[i, args.maskweight])
            #     if o!=args.ords_use[0]:  #rest of the ords keep red region overlap, discard blue
            #         for i in datafit[o].index[cond[o]]:
            #             if datafit[o].at[i, 'cen'] > max(datafit[o-1]['cen']):
            #                 wm_join.append(datafit[o].at[i, args.maskwave])
            #                 fm_join.append(datafit[o].at[i, args.maskweight])

    else:
        wm_join = datafit_final[0][args.maskwave]
        fm_join = datafit_final[0][args.maskweight]

    wm_join = np.array(wm_join)
    fm_join = np.array(fm_join)

    verboseprint('\n-> Total number of lines: {}\n'.format(len(wm_join)))

    ###############################################################################

    # Shift mask tplrv
    # ----------------

    verboseprint('\nMask to absolute RV: Shift mask by `tplrv`, the RV of the star: RVstar = {:.3f} km/s (input user or Carmencita)\n  w_abs = w * (1 - RVstar / c)'.format(tplrv))

    wm_join = wm_join*(1-tplrv/C_KMS)

    # CCF PHOENIX, shift mask
    # -----------------------

    # TODO

    if (args.filphoenixw is not None) or (args.filphoenixf is not None):
        verboseprint('\nMask to absolute RV refine: Compute CCF with PHOENIX broadened spectrum to refine the absolute RV measurement')

        # Get PHOENIX files
        # - Wavelength
        if not os.path.isfile(args.filphoenixw):
            sys.exit('PHOENIX w file not found: {}'.format(args.filphoenixw))

        # - Flux
        # -- Select file from provided ones
        if args.filphoenixf == 'select':
            sys.exit('Not implemented yet!')
            # try:
            #     spt = datacarmencita['SpTnum'].loc[args.obj]
            #     filphoenixf = ccflib.selectfilphoenixf(spt)
            # except:
            #     sys.exit('  Cannot find SpT of {} in carmencita'.format(args.obj))

        else:
            if not os.path.isfile(args.filphoenixf):
                sys.exit('PHOENIX f file not found: {}'.format(args.filphoenixf))

        # Load PHOENIX
        wp, fp = phoenixutils.read_phoenixfits(args.filphoenixw, args.filphoenixf)
        # Cut PHOENIX
        if args.phoenixwmin is None or args.phoenixwmax is None:
            if args.inst == 'CARM_VIS':
                wslice1 = 5600.
                wslice2 = 8700.
            else:
                wslice1 = np.nanmin(wm_join) - np.nanmin(wm_join) * 10. / C_KMS
                wslice2 = np.nanmax(wm_join) + np.nanmax(wm_join) * 10. / C_KMS
        else:
            wslice1 = args.phoenixwmin
            wslice2 = args.phoenixwmax
        wp, fp = phoenixutils.slice_phoenix(wslice1, wslice2, wp, fp, verb=args.verbose)
        wp_raw, fp_raw = wp, fp

        # ------------------

        # Instrumental broadening
        verboseprint('  Instrumental broadening')
        if args.phoenixbroadinstR is not None:
            # Get value
            if args.phoenixbroadinstR == 'inst':
                R = spectrographutils.dicres[args.inst]
            elif pyutils.isfloatnum(args.phoenixbroadinstR):
                R = float(args.phoenixbroadinstR)
            else:
                sys.exit('args.phoenixbroadinstR = {}, not valid. Exit.'.format(args.phoenixbroadinstR))
            # Run
            verboseprint('    R = {}'.format(R))
            fpi = spectrumutils.spec_conv_gauss_custom(wp, fp, resolution=R, dwindow=2, verb=args.verbose)
            fp = fpi.copy()

        # ------------------

        # # Rotational broadening -> Computed before. TODO integrate
        # verboseprint('  Rotational broadening')

        # if args.phoenixbroadrotV is not None:
        #     # - Get broaden velocity
        #     if args.phoenixbroadrotV == 'carmencita':
        #         V = datacarmencita['vsini_kms-1'].loc[args.obj]
        #         if not np.isfinite(V):
        #             sys.exit('Cannot find vsini of {} in carmencita'.format(args.obj))
        #     elif pyutils.isfloatnum(args.phoenixbroadrotV):
        #         V = float(args.phoenixbroadrotV)
        #     else:
        #         sys.exit('args.phoenixbroadrotV = {}, not valid. Exit.'.format(args.phoenixbroadrotV))
        #     verboseprint('    Rotational velocity: {}'.format(V))

        #     # - Lin limb-darkening
        #     limbdark = 0.

        #     # - Get broadened spectrum
        #     # -- Use all spectrum ---> Slow!!
        #     fpib = ...
        #     fp = fpib.copy()

            # rvrng = V

        # # If no rotational broadening given/applied, assume narrow lines -> narrow CCF
        # else:
        #     V = 0.
        #     if args.tplrv is None: rvrng = 50.
        #     else: rvrng = 5.

        if args.tplrv is None: rvrng = 50.
        else: rvrng = 5.

        # ------------------

        # Compute CCF
        verboseprint('  Compute CCF')

        cp = np.ones_like(wp)
        rvp = np.arange(-rvrng*3, rvrng*3, 0.5)
        ccfp, _ = ccflib.computeccf(wp, fp, cp, wm_join, fm_join, rvp, ron=None)

        # plt.plot(rvp, ccfp), plt.show(), plt.close()

        # Determine fit range: CCF maxima closest to absolute minimum
        # - CCF minima and maxima
        #    Mask nans
        masknan = np.isfinite(ccfp)
        limin, limax1, limax2 = peakutils.find_abspeaks(ccfp[masknan], method='custom')
        # - Maxima closest to CCF minimum
        iminccf = np.nanargmin(ccfp[masknan])
        i = np.where(limin == iminccf)[0][0]
        imax1ccf, imax2ccf = limax1[i], limax2[i]
        # Handle array ends
        if imax2ccf < len(ccfp[masknan]): imax2ccfp = imax2ccf + 1  # imax2 plus 1
        else: imax2ccfp = imax2ccf
        # Fit Gaussian
        x = rvp[masknan][imax1ccf:imax2ccfp]
        y = ccfp[masknan][imax1ccf:imax2ccfp]
        lmfitresult = peakutils.fit_gaussian_peak(x, y, amp_hint=np.nanmin(y) - np.nanmax(y), cen_hint=rvp[masknan][iminccf], wid_hint=1., shift_hint=np.nanmax(y), minmax='min')

        fitpar = {}
        for p in lmfitresult.params.keys():
            if lmfitresult.params[p].value is not None: fitpar['fit'+p] = lmfitresult.params[p].value
            else: fitpar['fit'+p] = np.nan
            if lmfitresult.params[p].stderr is not None: fitpar['fit'+p+'err'] = lmfitresult.params[p].stderr
            else: fitpar['fit'+p+'err'] = np.nan
        fitpar['fwhm'] = peakutils.gaussian_fwhm(wid=fitpar['fitwid'])
        fitpar['fitredchi2'] = lmfitresult.redchi

        verboseprint('Fit Gaussian to CCF')
        verboseprint('  CCF RV: {:.2f} +- {:.2f} m/s'.format(fitpar['fitcen']*1.e3, fitpar['fitcenerr']*1.e3))
        verboseprint('  CCF FWHM: {:.2f} m/s'.format(fitpar['fwhm']*1.e3))  # , fitpar['fwhmerr']*1.e3))

        # # Plot
        # fig, ax0, ax1 = ccflib.plot_ccf_fit_diff(rvp, ccfp, fitpar, title='CCF Mask + PHOENIX {}\nInstr R={:.0f}, Rot {:.2f} km/s'.format(os.path.basename(args.filphoenixf)[:17], R, V), fitpar='', diffzoom=True, parvelunit='ms')
        # plotutils.figout(fig, filout=os.path.join(args.dirout_plot, 'phoenix_broadR{:.0f}V{:.0f}_ccf'.format(R, V)), sv=True, svext=args.plot_ext, sh=args.plot_sh)

        # -----------------------------

        # Final shift mask

        # Shift mask
        verboseprint('Shift between mask and PHOENIX: RVccf = {:.2f} m/s'.format(fitpar['fitcen']*1.e3))
        verboseprint('Shift mask by this value w_abs = w * (1 - RVccf / c)')
        wm_join = wm_join*(1-fitpar['fitcen']/C_KMS)

    # Save mask
    # ---------

    # Data
    dataout = np.stack((wm_join, fm_join))

    # Output file name
    if args.filoutmask is None:
        filoutmask = os.path.basename(os.path.splitext(args.filtpl)[0])+'.mas'

    elif args.filoutmask == 'cuts':
        tpl = '{}{}'.format(args.obj, args.tpltype)
        # tell = '_tell{:5.0f}'.format(tellbroadendv) if args.filtell is not None else ''
        if args.filtell is not None:
            if args.tellbroadendv == 'obsall':
                tell = '_tellbervmax'
            else:
                tell = '_tell{:.0f}'.format(tellbroadendv)
        else:
            tell = ''
        cuts = '_fwhm{:2.2f}-{:2.2f}_contrminmin{:.2f}_depthwq{:.2f}'.format(args.line_fwhmmin, args.line_fwhmmax, args.line_contrastminmin, args.line_depthw_depthmaxquantile)
        maskweight = '_{}'.format(args.maskweight)
        filoutmask = '{}{}{}{}.mas'.format(tpl, tell, cuts, maskweight)
    else:
        filoutmask = args.filoutmask

    verboseprint('\nSave mask data in:', os.path.join(args.dirout_mask, filoutmask))
    np.savetxt(os.path.join(args.dirout_mask, filoutmask), dataout.T, fmt='%.10f')

    # Save processed template shifted to RV=0
    # ---------------------------------------

    if (args.filphoenixw is not None) or (args.filphoenixf is not None):
        w_shift = [i * (1 - (tplrv + fitpar['fitcen']) / C_KMS) for i in w]
        filout = os.path.join(args.dirout_tpl, os.path.splitext(os.path.basename(args.filtpl))[0] + 'matrix_rv0.pkl')
        spectrumutils.spec_save_pkl_matrix(w_shift, f, filout, verb=args.verbose)

    # Save clean minima wavelengths shifted fit data
    # ----------------------------------------------

    if (args.filphoenixw is not None) or (args.filphoenixf is not None):

        for o in ords:
            if o in args.ords_use and len(imin[o] > 0):

                # Shift
                lisp = ['cen', 'wmin', 'wmax1', 'wmax2']
                for p in lisp:
                    datafit[o][p] = datafit[o][p] * (1 - (tplrv + fitpar['fitcen']) / C_KMS)

                # Save
                colp = ['cen', 'wmin', 'wmax1', 'wmax2', 'depth']
                fil = os.path.join(args.dirout_minfit, 'minfito{:02d}_w_clean_rv0.dat'.format(o))
                datafit[o].to_csv(fil, sep=' ', na_rep=np.nan, float_format=None, columns=colp, header=True)

            else:
                # Save empty file
                fil = os.path.join(args.dirout_minfit, 'minfito{:02d}_w_clean_rv0.dat'.format(o))
                os.system('touch {}'.format(fil))


    return

###############################################################################


if __name__ == "__main__":

    print('Running:', sys.argv)

    main(sys.argv[1:])

    print('End')
