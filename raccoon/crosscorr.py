"""
Cross-correlation class and functions.
"""
import os
from pathlib import Path
import sys

from astropy.io import fits
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from popurri import plotutils
from popurri.plotutils import wavelength_label
from popurri import spectrum
from popurri import telluricutils

from . import ccflibfort
from . import peakutils
# from . import telluricutils

# dirhere = os.path.dirname(__file__)
# dirdata = os.path.join(dirhere, './data/')

###############################################################################

# Model utils
# -----------

def get_obj_info():
    """
    Get object absolute RV, spectral type, and vsini.
    - absolute RV and vsini used to determine CCF shift and range.
    - spectral type (and vsini, but not implemented yet!) used to select one of the default masks (only available for some instruments).
    """
    obj = 'obj'
    return obj


def show_available_models():
    """
    Print avaialble default models (for instrument, spectral type, vsini).
    """
    # TODO
    return dfmodels


def find_closest_model(inst, spt, vsini):
    """
    Given the spectral type and vsini of the object, and the instrument of the data, find the closest default model of the ones available in raccoon.

    Returns
    -------
    filmodel : str
        File with the model.
    keymodel : str
        Key to access the model in the dictionary/dataframe.
    modeltype : str
        Type of model (e.g. mask).
    """
    # TODO
    return filmodel, keymodel, modeltype


# =============================================================================

# Model class
# -----------

class Model():
    """
    CCF Model class
    
    Binary mask or full model (e.g. observed or synthetic spectrum) to cross-correlate with the data.

    NOTE: Only binary mask implemented so far.
    """
    def __init__(self, filin, modeltype='mask', maskformatharp=False, maskair=False, tag=None, dirout='./'):
        """

        datamod : dict with keys 'wm', 'fm', 'nlin', 'ords', 'nord'
        datamod['wm'] will be returned in vacuum.

        Parameters
        ----------
        filin : str
            Can be:
            - input file with the model (can be one of the default models, selected with `find_closes_model`)
            - popurri.spectrum.Spectrum object if modeltype is `Spectrum`
        modeltype : 'mask' (default), 'phoenix', 'Spectrum', 'fil1D'
            Type of model. Default is 'mask'.
            TODO: Explain types.
        
        Returns
        -------
        datamod : dict
            Mask data in dictionary form. Keys: wm, fm, nlin, ords, nord

        """
        self.filin = filin
        self.modeltype = modeltype
        self.maskformatharp = maskformatharp
        self.maskair = maskair
        self.tag = tag
        self.utag = '_' + tag
        self.dirout = dirout
        if not os.path.exists(self.dirout): os.makedirs(self.dirout)

        # Read model from `filin`
        self.datamod = self.read_model()


    def read_model(self):
        """
        maskformatharp : bool, default=False
            HARPS DRS masks have 3 columns: wm1, wm2, fm, where wm1 and wm2 are in air. If True, the mask is assumed to have this format when reading it, then a single wm is obtained from the mean of wm1 and wm2, and wm is converted from air to vacuum.
        maskair : bool, default=False
            The wm of the model is assumed to be in vaccum, as opposed to air. If True, the wm is converted from air to vacuum.
        """
        # Read model
        if self.modeltype == 'mask':
            if self.maskformatharp:
                wm1, wm2, fm = np.loadtxt(self.filin, usecols=[0, 1, 2], unpack=True)
                wm = (wm1 + wm2) / 2.
                wm = spectrum.wair2vac(wm)  # Applies air2vac correction even if not specified with maskair=True
                self.maskair = False
            else:
                wm, fm = np.loadtxt(self.filin, usecols=[0, 1], unpack=True)
            nlin = len(wm)
            ords = np.array([0])
            nord = 1

        elif self.modeltype == 'phoenix':
            print('To be implemented.')
            sys.exit(1)
            # TODO

        elif self.modeltype == 'Spectrum':
            # filin should be the a Spectrum object (so already read and stored in memory)
            wm = self.filin.dataspec['w']
            fm = self.filin.dataspec['f']
            nlin = self.filin.dataspec['w'].shape
            ords = self.filin['ords']
            nord = self.filin['nord']

        elif self.modeltype == 'fil1D':
            wm, fm = np.loadtxt(self.filin, usecols=[0, 1], unpack=True)
            nlin = len(wm)
            ords = np.array([0])
            nord = 1

        else:
            print('Model type not recognized.')
            sys.exit(1)

        # Convert from air to vacuum
        if self.maskair:
            wm = spectrum.wair2vac(wm)
            self.maskair = False

        datamod = {'wm': wm, 'fm': fm, 'nlin': nlin, 'ords': ords, 'nord': nord}
        return datamod


    def plot_mask(self, ax=None, ymin=0, yscale=1, xunit='A', xlabel=None, ylabel='Mask weight', leglabel='', title='', color='0.5', alpha=0.8, linestyles='solid', zorder=0, keeplims=False, **kwargs):
        """Plot binary mask"""
        if ax is None: ax = plt.gca()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.vlines(self.datamod['wm'], ymin, self.datamod['fm']*yscale, color=color, linestyles=linestyles, alpha=alpha, zorder=zorder, label=leglabel, **kwargs)
        if keeplims: ax.set(xlim=xlim, ylim=ylim)
        if xlabel is None: xlabel = wavelength_label(x=xunit)  # Assumes plotting wavelength in x-axis
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        return ax


    def fig_mask(self, figsize=(16, 4), filout=None, sh=False, sv=True, svext=['pdf'], **kwargs):
        """
        """
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax = self.plot_mask(ax=ax, **kwargs)
        if filout is None:
            filout = os.path.join(self.dirout, 'mask' + self.utag)
        plotutils.figout(fig, sv=sv, filout=filout, svext=svext, sh=sh)
        return

###############################################################################

# CCF utils
# ---------

def computeccf(w, f, c, wm, fm, rv, ron=None, forig=None):
    """
    Wrapper for the fortran functions `ccflibfort.computeccf`, `ccflibfort.computeccferr`, `computeccferrfluxorig`.

    Compute the CCF error only if `ron` is not None (default). Not computing the error makes the execution slightly faster.

    Parameters
    ----------
    w, f : 1d array like, same shape
        Spectrum (1d, i.e. for 1 order). Wavelength units must be consistent with mask wavelength `wm`.
    c : 1d array like, same shape as `w` and `f`
        Blaze function, to take into account the observed flux. If no blaze, input an array with ones, e.g. `c = np.ones_like(w)`.
    wm, fm : 1d array like, same shape
        Mask position (`wm`) and weight (`fm`).
    rv : 1d array-like
        RV array for which the mask is Doppler-shifted.
    ron : float
        Read out noise. If None (default), no CCF error is computed.
    forig : 1d array
        Original spectrum flux (before correcting for order ratios).

    Returns
    -------
    ccf, ccferr : 1d array
    """

    # Check non-empy arrays
    if len(w) == 0 or len(f) == 0 or len(c) == 0 or len(wm) == 0 or len(fm) == 0:
        ccf, ccferr = rv * np.nan, rv * np.nan
        return ccf, ccferr

    if ron is None:
        ccf = ccflibfort.computeccf(w, f, c, wm, fm, rv)
        ccferr = np.ones_like(ccf) * np.nan

    else:
        if forig is None:
            ccf, ccferr = ccflibfort.computeccferr(w, f, c, wm, fm, rv, ron)
        else:
            ccf, ccferr = ccflibfort.computeccferrfluxorig(w, f, c, forig, wm, fm, rv, ron)
    return ccf, ccferr


def computerverr(rv, ccf, ccferr):
    """Compute RV error from CCF profile.
    """
    der = np.ones_like(rv) * np.nan
    # rverr = np.ones_like(rv) * np.nan
    rverrsum = 0.
    for i in range(len(rv)):

        # Derivative
        if i == 0:  # start
            der[i] = np.abs((ccf[i+1]-ccf[i])/(rv[i+1]-rv[i]))
        elif i == len(rv)-1:  # end
            der[i] = np.abs((ccf[i]-ccf[i-1])/(rv[i]-rv[i-1]))
        else:  # rest
            der[i] = np.abs(((ccf[i+1]-ccf[i])/(rv[i+1]-rv[i]) + (ccf[i]-ccf[i-1])/(rv[i]-rv[i-1]))/2.)

    ccferr = np.array(ccferr)
    der = np.array(der)

    # RV err
    rverr = ccferr / der

    # RV err total
    rverrsum = np.sum(1. / rverr**2)
    rverrt = 1./np.sqrt(rverrsum)
    # return rverrt
    return rverrt, der, rverr


def computebisector(x, y, xerr, n=100):
    """
    Compute bisector and its errors.

    Parameters
    ----------
    x, y : 1d arrays
        Data. Must have a Gaussian-like shape.
    xerr : 1d array
        x datapoints errors.
    n : int (default 100)
        Number of points of bisector.

    Returns
    -------
    bx, by : 1d arrays
        Bisector x and y coordinates.
    bxerr : 1d array
        Bisector x datapoints error.
    bx1, bx2 : 1d arrays
        x values at the bisector heights `by` for each side of the line.
    bx1err, bx2err : 1d arrays
        Errors for the data `bx1` and `bx2`.

    """

    # y minimum and maxima (maxima: absolute maxima each side)
    imin = np.nanargmin(y)  # Minimum
    imax1 = np.nanargmax(y[:imin])  # Maximum left part
    imax2 = imin + np.nanargmax(y[imin:])  # Maximum right part
    if imax2 == len(y): imax2p = imax2
    else: imax2p = imax2 + 1  # plus one

    y_smallestmax = np.nanmin([y[imax1], y[imax2]])  # Smallest maximum

    # Bisector y heights
    by = np.linspace(y[imin], y_smallestmax, n)

    # Interpolate bisector y to x for both sides of the y
    #  interp1d(x, y)
    # - Function
    interpolate_x1 = interp1d(y[imax1:imin+1], x[imax1:imin+1], kind='linear')
    interpolate_x2 = interp1d(y[imin:imax2p], x[imin:imax2p], kind='linear')
    # - Bisector x values
    bx1 = interpolate_x1(by)
    bx2 = interpolate_x2(by)

    # Compute bisector
    bx = (bx2 + bx1)/2.

    # -----------------------

    # Bisector error

    # xerr
    # Do not have RVerr for the bisector x datapoints, only for the original x points
    # Solution: Interpolate the error
    # - Function
    interpolate_x1err = interp1d(y[imax1:imin+1], xerr[imax1:imin+1], kind='linear')
    interpolate_x2err = interp1d(y[imin:imax2p], xerr[imin:imax2p], kind='linear')
    # - Bisector x error values
    bx1err = interpolate_x1err(by)
    bx2err = interpolate_x2err(by)

    # Compute bisector error (error propagation)
    bxerr = np.sqrt(bx1err**2 + bx2err**2) / 2.

    return bx, by, bxerr, bx1, bx2, bx1err, bx2err


def computebisector_bis(x, y, ybotmin_percent=10., ybotmax_percent=40., ytopmin_percent=60., ytopmax_percent=90., verb=True):
    """
    Compute bisector inverse slope (BIS).

    Note that the bisector is not interpolated in order to find the points closest to the region limits defined by `ybotmin_percent` etc. So if bisector sampling is low the points won't be correctly selected.

    Parameters
    ----------
    x, y : 1d array like
        Bisector coordinates.
    ybotmin_percent, ybotmax_percent : float
        Bisector bottom region limits in percentage.
    ytopmin_percent, ytopmax_percent : float
        Bisector top region limits in percentage.

    Notes
    -----
    BIS definition from Queloz et al. 2001
    """

    # Check bisector sampling
    s = len(y)
    warn = 'Not good sampling to compute BIS!' if s < 100. else ''
    if verb: print('  {} points in bisector.', warn)

    # Bisector up and down region limits -> absolute value
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    y_delta = y_max - y_min

    ybotmin = y_min + y_delta * ybotmin_percent/100.
    ybotmax = y_min + y_delta * ybotmax_percent/100.
    ytopmin = y_min + y_delta * ytopmin_percent/100.
    ytopmax = y_min + y_delta * ytopmax_percent/100.

    # Bisector up and down region limits indices
    # Note: Approximate regions, depend on bisector sampling
    iybotmin = np.nanargmin(np.abs(y-ybotmin))
    iybotmax = np.nanargmin(np.abs(y-ybotmax))
    iytopmin = np.nanargmin(np.abs(y-ytopmin))
    iytopmax = np.nanargmin(np.abs(y-ytopmax))

    # Compute mean RV in each region
    #### Should it be a weighted mean?
    xmeantop = np.nanmean(x[iytopmin:iytopmax+1])
    xmeanbot = np.nanmean(x[iybotmin:iybotmax+1])

    # Compute bisector inverse slope BIS
    bis = xmeantop - xmeanbot

    return bis, xmeantop, xmeanbot, iybotmin, iybotmax, iytopmin, iytopmax


def computebisector_biserr(x, y, xerr, n=100, bybotmin_percent=10., bybotmax_percent=40., bytopmin_percent=60., bytopmax_percent=90., xrealsampling=None, verb=True, returnall=False):
    """
    Compute bisector, bisector inverse slope (BIS) and their errors.

    Same code as in functions `computebisector` to compute the bisector and in `computebisector_bis` to compute the BIS. Copied here because intermediate products needed to compute the BIS error.

    If do not need the BIS error, use the other functions
    - `computebisector`
    - `computebisector_bis`

    Parameters
    ----------

    x, y : 
    xerr : 
    n : int (default 100)
    bybotmin_percent, bybotmax_percent : float
        Bisector bottom region limits in percentage.
    bytopmin_percent, bytopmax_percent : float
        Bisector top region limits in percentage.
    xrealsampling : float (default None)
        Sampling of the real data. Usually the input data provided (`x` and `y`) is oversampled in order to correctly compute the bisector. So need to provide what would be the real sampling in order to compute the BIS error properly.
    bybotmin_percent ...
    verb
    returnall : bool (default False)
        Output a lot intermediate of products.

    Returns
    -------

    """

    # Bisector
    # --------

    # y minimum and maxima (maxima: absolute maxima each side)
    imin = np.nanargmin(y)  # Minimum
    imax1 = np.nanargmax(y[:imin])  # Maximum left part
    imax2 = imin + np.nanargmax(y[imin:])  # Maximum right part
    if imax2 == len(y): imax2p = imax2
    else: imax2p = imax2 + 1  # plus one

    y_smallestmax = np.nanmin([y[imax1], y[imax2]])  # Smallest maximum

    # Bisector y heights
    by = np.linspace(y[imin], y_smallestmax, n)

    # Interpolate bisector y to x for both sides of the y
    #  interp1d(x, y)
    # - Function
    interpolate_x1 = interp1d(y[imax1:imin+1], x[imax1:imin+1], kind='linear')
    interpolate_x2 = interp1d(y[imin:imax2p], x[imin:imax2p], kind='linear')
    # - Bisector x values
    bx1 = interpolate_x1(by)
    bx2 = interpolate_x2(by)

    # Compute bisector
    bx = (bx2 + bx1)/2.

    # -----------------------

    # Bisector error
    # --------------

    # xerr
    # Do not have RVerr for the bisector x datapoints, only for the original x points
    # Solution: Interpolate the error
    # - Function
    interpolate_x1err = interp1d(y[imax1:imin+1], xerr[imax1:imin+1], kind='linear')
    interpolate_x2err = interp1d(y[imin:imax2p], xerr[imin:imax2p], kind='linear')
    # - Bisector x error values
    bx1err = interpolate_x1err(by)
    bx2err = interpolate_x2err(by)

    # Compute bisector error (error propagation)
    bxerr = np.sqrt(bx1err**2 + bx2err**2) / 2.

    # -----------------------

    # BIS
    # ---

    # Check bisector sampling
    s = len(by)
    warn = 'Not good sampling to compute BIS!' if s < 100. else ''
    if verb: print('  {} points in bisector.'.format(s), warn)

    # Bisector up and down region limits -> absolute value
    by_min = np.nanmin(by)
    by_max = np.nanmax(by)
    by_delta = by_max - by_min

    bybotmin = by_min + by_delta * bybotmin_percent/100.
    bybotmax = by_min + by_delta * bybotmax_percent/100.
    bytopmin = by_min + by_delta * bytopmin_percent/100.
    bytopmax = by_min + by_delta * bytopmax_percent/100.

    # Bisector up and down region limits indices
    # Note: Approximate regions, depend on bisector sampling
    ibybotmin = np.nanargmin(np.abs(by-bybotmin))
    ibybotmax = np.nanargmin(np.abs(by-bybotmax))
    ibytopmin = np.nanargmin(np.abs(by-bytopmin))
    ibytopmax = np.nanargmin(np.abs(by-bytopmax))

    # Compute mean RV in each region
    #### Should it be a weighted mean?
    bxmeantop = np.nanmean(bx[ibytopmin:ibytopmax+1])
    bxmeanbot = np.nanmean(bx[ibybotmin:ibybotmax+1])

    # Compute bisector inverse slope BIS
    bis = bxmeantop - bxmeanbot

    # -----------------------

    # BIS error
    # ---------

    # Compute number of points in top and bottom regions
    #   Use regions x width and x sampling

    # - Original data x sampling
    if xrealsampling is None:
        dx_data = x[1] - x[0]
        if verb:
            print('No real sampling provided for bisector! Need to compute BIS error. Using sampling from input data: {}'.format(dx_data))
    else:
        dx_data = xrealsampling

    # - Top and bottom regions x width (use only one side of the line)
    dx_top = np.abs(interpolate_x1(bytopmax) - interpolate_x1(bytopmin))
    dx_bot = np.abs(interpolate_x1(bybotmax) - interpolate_x1(bybotmin))

    # - Number of points
    ntop = dx_top / dx_data
    nbot = dx_bot / dx_data

    # Mean error top and bottom regions
    bxtopmeanerr = np.nanmean(bxerr[ibybotmin:ibybotmax+1]) / np.sqrt(ntop)
    bxbotmeanerr = np.nanmean(bxerr[ibytopmin:ibytopmax+1]) / np.sqrt(nbot)

    # BIS error
    biserr = np.sqrt(bxtopmeanerr**2 + bxbotmeanerr**2)

    if not returnall:
        return bx, by, bxerr, bis, biserr
    else:
        return bx, by, bxerr, bis, biserr, bx1, bx2, bx1err, bx2err, bxmeantop, bxmeanbot, ibybotmin, ibybotmax, ibytopmin, ibytopmax


# =============================================================================

# CCF class
# ---------

class Crosscorr():
    """
    Crosscorr class

    Parameters
    ----------
    data : popurri.spectrum.Spectra
        Observed spectra time series.
    model : Model
        Mask or model to cross-correlate with the data.
    tellmask : popurri.telluric.Mask
        Telluric mask to remove model lines affected by tellurics.

    obsref : str
        Observation (filename) to be used as reference.
    ccftestrun : bool, default True
        Only applies to `run_full_ccf`. If True, the initial test CCF is computed to find the adequate centre and range of the CCF.
    
    rvcen, rvrng, rvstp : float
    

    """
    def __init__(self, data, model, tellmask=None, dirout='./', tag=None,
        ords_use=None,
        bervmax=None,
        obs_ref=None,
        verbose=True,
        # Initial test CCF parameters
        ccftestrun=True, 
        ):
        self.data = data  # popurri.spectrum.Spectra
        self.model = model  # Model
        self.tellmask = tellmask
        self.ords_use = ords_use if ords_use is not None else self.data.ords
        self.tag = tag
        self.dirout = dirout
        if not os.path.exists(self.dirout): os.makedirs(self.dirout)

        # Initial test CCF parameters
        # Only initialise to None, they will be set in `computeccf_initial_test`, if run
        self.test_run = None
        self.test_rvcen = None
        self.test_rvrng = None
        self.test_rvstp = None
        self.test_o = None
        self.test_dmin = None
        self.test_rv = None
        self.test_ccf = None
        self.test_rvcen_result = None
        self.test_rvrng_result = None
        self.test_fitpar = None

        # CCF parameters
        # Only initialise to None, they will be set in `computeccf`, if run
        self.rvcen = None
        self.rvrng = None
        self.rvstp = None
        self.rvstpreal = None
        self.fitfunc = None
        self.fitrng = None
        self.fitrngeach = None
        self.fluxratiotype = 'obshighsnr'
        self.fluxratio = None
        self.nlinmin = 0

        # Select obs with highest S/N
        self.obs_ref = obs_ref
        if self.obs_ref is None:
            self.obs_ref = self.data.dataheader[f'snro{self.data.ord_ref}'].idxmax()
            # self.obs_ref = 0

        # If the model has more than 1 "order", check if data and model have the same number orders (they should)
        if self.model.nord > 1:
            if self.data.ords != self.model.ords:
                print('Data and model have different number of orders.')
                sys.exit(1)
        

    # =========================================================================

    # CCF preparation functions
    # -------------------------

    def compute_ord_fluxratio(self, fluxratiotype='obshighsnr', save=True):
        """
        Get order relative flux correction

        Flux correction obtained from reference observation `obs_ref`, so that the flux of the  orders of all the observations always have the same ratio.
        Returns the flux ratios per order (all orders, not only those in ords_use), to be used as weights in the CCF computation.
        """
        self.fluxratiotype = fluxratiotype
        if self.fluxratiotype == 'obshighsnr':
            
            # Reference order
            oref = self.data.ord_ref

            # Make sure reference obs is the one with highest S/N
            obssnrmax = self.data.dataheader[f'snro{oref}'].idxmax()
            iobs = self.data.dataheader.index.get_loc(obssnrmax)

            # Flux of reference obs
            f = self.data.dataspec['f'][iobs]
            # c = self.data.dataspec['c'][self.obs_ref][self.test_o]

            # TODO Revise divide by cont, or not
            # TODO Revise normalise flux to snr, or not
            # # Cont
            # f = [f[o]/c[o] for o in ords]
            # # fluxratio2 = [np.nanmedian(f[o]) / np.nanmedian(f[oref]) for o in ords]
            # # SNR
            # if args.inst == 'CARM_VIS' or args.inst == 'CARM_NIR':
            #     f = [f[o] * dataobs.loc[filobsref]['snro{:d}'.format(o)]**2 / np.nanmedian(f[o]) for o in ords]
            ##### NEW
            ##### lissnro = [self.data.dataheader.loc[self.obs_ref, f'snro{o}'] for o in self.ords]

            fluxratio = [np.nanmedian(f[o]) / np.nanmedian(f[oref]) for o in self.ords]

        else:
            # Set all ratios to 1
            fluxratio = np.ones_like(self.ords)

        self.fluxratio = fluxratio
        # Save flux ratios per order: Order, flux ratio (i.e. weight), lambda min order, lambda max order
        if save:
            w = self.data.dataspec['w'][iobs]
            arrwmin = [np.nanmin(w[o]) for o in self.ords]
            arrwmax = [np.nanmax(w[o]) for o in self.ords]
            filout = os.path.join(args.dirout, f'{args.obj}.fluxratio.csv')
            np.savetxt(filout, np.vstack((self.ords, fluxratio, arrwmin, arrwmax)).T, delimiter=',', fmt=['%d', '%.8f', '%.8f', '%.8f'])
            return


    def select_masklines_telloverlap(self):
        pass
        return

    def select_masklines_freetellurics(self):
        pass
        return
    


    # =========================================================================

    # CCF computation functions
    # -------------------------

    def computeccf_initial_test(self, test_rvcen=0., test_rvrng=200., test_rvstp=1., test_o=None, ccftestdmin=2,):
        """
        Compute the cross-correlation on a reference order (the ord_ref of the instrument by default), on a reference observation (the one with highest S/N by default), to find the adequate centre and range of the CCF, if not provided with `rvcen` and `rvrng`.

        test_rvcen_result
        test_rvrng_result
        """
        # Inputs
        self.test_rvcen = test_rvcen
        self.test_rvrng = test_rvrng
        self.test_rvstp = test_rvstp
        self.test_o = test_o if test_o is not None else self.data.ord_ref
        self.test_dmin = ccftestdmin
        # Outputs
        self.test_rv = None
        self.test_ccf = None
        self.test_rvcen_result = None
        self.test_rvrng_result = None
        self.test_fitpar = None

        if self.verbose: print(f'Computing initial test CCF on order {self.test_o}, RV center {self.test_rvcen}, RV range +- {self.test_rvrng}, and RV step {self.test_rvstp}, on observation {self.obs_ref}, {self.data.lisfil[self.obs_ref]}')

        w = self.data.dataspec['wcorr'][self.obs_ref][self.test_o]
        f = self.data.dataspec['f'][self.obs_ref][self.test_o]
        c = self.data.dataspec['c'][self.obs_ref][self.test_o]
        wm = self.model.datamod['wm']
        fm = self.model.datamod['fm']
        self.test_rv = np.arange(self.test_rvcen-self.test_rvrng, self.test_rvcen+self.test_rvrng+self.test_rvstp, self.test_rvstp)

        # Compute CCF test
        self.test_ccf = ccflibfort.computeccf(w, f, c, wm, fm, self.test_rv)

        # Find CCF centre (minimum)
        imin = np.nanargmin(self.test_)
        test_rvcen_result = self.test_rv[imin]
        if self.verbose: print('  RV cen: {} km/s'.format(test_rvcen_result))

        # Fit Gaussian to estimate CCF range

        # - Determine fit range: CCF maxima closest to absolute minimum (helps with noisy CCFs)
        # -- CCF absolute minima and maxima
        mnan = np.isfinite(self.test_ccf)  # Mask nans
        rv, ccf = self.test_rv[mnan], self.test_ccf[mnan]
        limin, limax1, limax2 = peakutils.find_abspeaks(ccf, method='custom')
        # - Maxima closest to CCF minimum
        imin = np.nanargmin(ccf)
        i = np.where(limin == imin)[0][0]
        imax1, imax2 = limax1[i], limax2[i]
        # - Handle array ends
        if imax2 < len(ccf): imax2p = imax2 + 1
        else: imax2p = imax2

        # - Check that the distance in RV between maxima is at least self.test_dmin (2 km/s by default)
        # If not, select next maxima until it happens
        # If never happens, will fit all RV range
        # Helps with noisy CCFs
        i1, i2 = i, i
        while rv[imax2p] - rv[imax1] < self.test_dmin:
            # If reached end of RV array, the fit limits are all the range
            if imax1 == limax1[0] or imax2p == limax2[-1]:
                imax1 = limax1[0]
                imax2p = limax2[-1]
                if self.verbose: print(f'  Cannot constrain CCF minimum range. Using all range ({rv[imax1]} -- {imax2p}) to fit a Gaussian and determine width.')
                break
            # Go to next closest maxima at each side
            i1 = i1 - 1
            i2 = i2 + 1
            imax1, imax2 = limax1[i1], limax2[i2]
            if imax2 != limax2[-1]: imax2p = imax2 + 1
            else: imax2p = imax2

        # - Fit Gaussian
        x = rv[imax1:imax2p]
        y = ccf[imax1:imax2p]
        lmfitresult = peakutils.fit_gaussian_peak(x, y, amp_hint=np.nanmin(y) - np.nanmax(y), cen_hint=rv[imin], wid_hint=1., shift_hint=np.nanmax(y), minmax='min')

        fitpar = {}
        for p in lmfitresult.params.keys():
            if lmfitresult.params[p].value is not None: fitpar['fit'+p] = lmfitresult.params[p].value
            else: fitpar['fit'+p] = np.nan
            if lmfitresult.params[p].stderr is not None: fitpar['fit'+p+'err'] = lmfitresult.params[p].stderr
            else: fitpar['fit'+p+'err'] = np.nan
        fitpar['fwhm'] = peakutils.gaussian_fwhm(wid=fitpar['fitwid'])
        fitpar['fitredchi2'] = lmfitresult.redchi

        # Determine CCF width: About 3 * FWHM
        test_rvrng_result = round(np.ceil(fitpar['fwhm']*3.))  # [km/s]
        # test_rvrng_result = round(np.ceil(fitpar['fwhm']*3.)/5)*5 # [km/s], round to 5
        if self.verbose: print(f'  RV range: {test_rvrng_result} km/s (FWHM {fitpar["fwhm"]}, chi2red {fitpar["fitredchi2"]})')

        # The Gaussian fit can give a more accurate minimum for the CCF, but can be problematic with noisy CCFs, so keep as rvcen_result the minimum found with np.argmin

        # Save results
        self.test_rvcen_result = test_rvcen_result
        self.test_rvrng_result = test_rvrng_result
        return


    def computeccf_lisobs(self,
        # CCF parameters
        rvcen=None, rvrng=None, rvstp=0.25, rvstpreal=None, use_test=False,
        fluxratiotype='obshighsnr', savefluxratio=True,
        nlinemin=0,
        # CCF fit parameters
        fitfunc='gaussian', fitrng='maxabs', fitrngeach=False,
        ):
        """
        Compute the cross-correlation between the observations in self.data and the model in self.model.

        The CCF is computed order-by-order, and the CCFs of all orders are coadded into a final CCF per observation.

        Parameters
        ----------

        use_test : bool, default False
            If True, use the results from the initial test CCF (test_rvcen_result` and `test_rvrng_result`) to set the CCF parameters, obtained with `computeccf_initial_test`.
            `computeccf_initial_test` must have been run before!
            Overrides `rvcen` and `rvrng`.
        """
        if self.verbose: print(f'Computing CCFs for {len(self.data.lisfil)} obs')

        # Set CCF parameters
        self.rvcen = rvcen if use_test and (self.test_rvcen_result is not None) else rvcen
        if use_test and (self.test_rvcen_result is None): print(f'  Warning: use_test=True but test_rvcen_result is None, using rvcen={rvcen} instead.')
        self.rvrng = rvcen if use_test and (self.test_rvrng_result is not None) else rvrng
        if use_test and (self.test_rvrng_result is None): print(f'Warning: use_test=True but test_rvrng_result is None, using rvrng={rvrng} instead.')
        self.rvstp = rvstp
        self.rvstpreal = rvstpreal if rvstpreal is not None else self.data.pixel_ms
        if rvstpreal is not None: print(f'  Warning: rvstpreal set by user to {rvstpreal}, however the correct value would be {self.data.pixel_ms}')
        self.fitfunc = fitfunc
        self.fitrng = fitrng
        self.fitrngeach = fitrngeach

        if self.rvstp >= self.rvstpreal: 
            if self.verbose: print(f'  Warning: RV step oversampled {self.rvstp} >= RV steps real {self.rvstpreal} -> May have problems with bisector')

        if self.verbose: print(f'CCF params:\n  RV center {self.test_rvcen}\n  RV range +- {self.test_rvrng}\n  RV step {self.test_rvstp}\n  ords {self.ords_use}\n  fitfunc {self.fitfunc}\n  fitrng {self.fitrng}\n  fitrngeach {self.fitrngeach}')

        # Final RV arrays
        rv = np.arange(self.rvcen - self.rvrng, self.rvcen + self.rvrng + self.rvstp, self.rvstp)
        rvreal = np.arange(self.rvcen - self.rvrng, self.rvcen + self.rvrng + self.rvstpreal, self.rvstpreal)
        
        # Compute order flux ratios
        compute_ord_fluxratio(fluxratiotype=fluxratiotype, save=savefluxratio)  # Set self.fluxratio

        # Select mask lines free of tellurics
        # TODO

        # Select mask lines usable at any epoch (BERV shift), per order
        # TODO

        # Remove orders with not enough lines based on nlinmin
        # TODO

        # Compute CCFs
        # TODO

        # TODO
        # Coadd order CCFs into final CCF per observation

        # Compute RV centre, FWHM, contrast by fitting a function (default Gaussian) to the CCF
        # TODO

        # Compute RV error
        # TODO


        # Compute bisector
        # TODO


        # Save results in files
        # TODO

        return


    def run_full_ccf(self):
        """
        Run full CCF computation.
        """
        # if self.test_run:
        #     self.computeccf_initial_test()

        # self.computeccf_lisobs()
        return


    def read_ccf(self):
        """Read previously computed raccoon CCFs from files"""
        # TODO
        pass
        return
    

    # =========================================================================

    # Plot functions
    # --------------

    def plot_ccf(self, filobs):
        """Plot CCF and fit of a single observation"""
        # TODO
        pass
        return

    def plot_ccfo(self, filobs):
        """Plot order CCFs of a single observation"""
        # TODO
        pass
        return

    def plot_ccfomap(self, filobs):
        """Plot map of order CCF of a single observation"""
        # TODO
        pass
        return
    
    def fig_ccf_ccfo_ccfomap(self, filobs):
        """Plot CCF (with fit), order CCFs and map of order CCFs of a single observation"""
        # TODO
        pass
        return

    def fig_diff_ccf_ccfo_ccfomap(self, filobs):
        """Same as `fig_ccf_ccfo_ccfomap`, but instead of the CCF, plot the CCF difference with respect to the CCF of a reference order"""
        # TODO
        pass
        return
    
    def plot_ccf_lisobs(self, lisfilobs):
        """Plot CCF  of all observations"""
        # TODO
        pass
        return
    
    def plot_ccfmap_lisobs(self, lisfilobs):
        """Plot map of CCFs of all observations"""
        # TODO
        pass
        return

    def fig_ccf_ccfmap_lisobs(self, lisfilobs):
        """Plot CCF and map of CCFs of all observations"""
        # TODO
        pass
        return

    def fig_diff_ccf_ccfmap_lisobs(self, lisfilobs, obs_ref=None):
        """Same as `fig_ccf_ccfmap_lisobs`, but instead of the CCF, plot the CCF difference with respect to the CCF of a reference observation (the one with highest S/N by default)
        """
        # TODO
        pass
        return

