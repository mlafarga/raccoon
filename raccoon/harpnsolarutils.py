# FITS data
# ---------


# Complete spectrum: w, f, blaze (optional) and headers

def drs_e2dsred_read(filin, readblaze=True, dirblaze=None, filblaze=None, inst=None, exte2ds=0, extb=0):
    """
    Read e2ds reduced spectrum flux and wavelength, and optionally the blaze.

    The wavelength data is obtained from the header keywords in `filin` using `drs_e2dsred_readw`.

    The blaze is obtained if `readblaze` is True.
    The directory containing the blaze files by default is the same as the directory where the e2ds file is (`filin`), but can be changed with `dirblaze`.
    The blaze file by default is obtained from the header of the e2ds file `filin`: 'HIERARCH TNG DRS BLAZE FILE', but can be changed with `filblaze`.

    Parameters
    ----------
    filin : str
        Reduced e2ds file.
    readblaze : bool, default True
        Whether to read the blaze or not. If False, the returned blaze `b` is an array full of ones of the same shape as the spectrum, and the header `header` is NaN.
    dirblaze : str, default None
        Directory where the blaze file is. If None (default), it is assumed that it is in the same directory as the spectrum `filin`.
    filblaze : str, default None
        Blaze file. Use if want to obtain the blaze from a file different than the one specified in the header keyword 'HIERARCH TNG DRS BLAZE FILE'.
    """

    # Read e2ds flux
    f, header = drs_fitsred_read(filin, ext=exte2ds)

    # Get wavelength from header
    # nord = len(f)
    npix = len(f[0])
    w = drs_e2dsred_readw(filin, inst=inst, npix=npix, ext=exte2ds)

    # Read blaze
    if readblaze:
        kwinst = headerkwinst(inst, outfail=np.nan)

        if dirblaze is None: dirblaze = os.path.dirname(filin)
        if filblaze is None: filblaze = header[kwinst + 'DRS BLAZE FILE']
        b, headerb = drs_fitsred_read(os.path.join(dirblaze, filblaze), ext=extb)
    else:
        b = np.ones_like(f)
        headerb = np.nan

    return w, f, b, header, headerb








