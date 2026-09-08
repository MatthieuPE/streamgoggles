import numpy as np

# You'll need to install jax: pip install -U jax
import logging
from types import SimpleNamespace
import healpy as hp
import numpy as np
from numpy.polynomial import polynomial
import pandas as pd


LOGGER = logging.getLogger(__name__)

#: A_band / E(B-V) extinction coefficients for the LSST bands, R_V = 3.1
#: (Schlafly & Finkbeiner 2011 scaling, as used in the Rubin DP0/DP1 tutorials).
#: Override per call if you prefer different coefficients.
EXTINCTION_COEFFS = {
    "u": 4.81, "g": 3.64, "r": 2.70, "i": 2.06, "z": 1.58, "y": 1.31,
}

#: This field is significantly reddened: E(B-V) ~ 0.57-0.88 (median 0.68) over 10 arcmin,
#: giving A_g ~ 2.46 mag and E(g-i) ~ 1.07 mag. Extinction correction is NOT optional
#: here, and because E(B-V) varies by 0.31 across the field (1.14 mag in A_g) it should
#: be applied per object rather than as a field mean.


def extinction(band, ebv, coeffs=None):
    """Extinction A_band in magnitudes for a given band and E(B-V)."""
    c = (coeffs or EXTINCTION_COEFFS)[band]
    return c * np.asarray(ebv, dtype=float)


def deredden(mag, band, ebv, coeffs=None):
    """Return extinction-corrected magnitude: ``mag - A_band``."""
    return np.asarray(mag, dtype=float) - extinction(band, ebv, coeffs=coeffs)


def deredden_dataframe(df, bands=("g", "r", "i"), ebv_col="ebv", coeffs=None,
                       mag_kinds=("psfMag",), suffix="_dered", inplace=False):
    """Add extinction-corrected magnitude columns to an object DataFrame.

    For each band and magnitude kind, adds ``{band}_{kind}{suffix}``, plus an
    ``A_{band}`` column recording the extinction applied. Uses the per-object ``ebv``
    column from the DP2 object table, which is the same dust map the DRP itself used.
    """
    out = df if inplace else df.copy()
    ebv = out[ebv_col].to_numpy(dtype=float)
    for b in bands:
        out[f"A_{b}"] = extinction(b, ebv, coeffs=coeffs)
        for kind in mag_kinds:
            col = f"{b}_{kind}"
            if col in out.columns:
                out[f"{col}{suffix}"] = out[col].to_numpy(dtype=float) - out[f"A_{b}"]
    return out


############################################################################################################
################################# Fitting method as in DEs paper ###########################################
############################################################################################################
# https://stackoverflow.com/a/32297563/4075339


def _validate_nside(nside):
    """Check that nside is a valid HEALPix resolution (power of 2)."""
    if not hp.isnsideok(nside):
        raise ValueError(f"Invalid nside={nside}. nside must be a power of 2.")


def process_data(data, **kwargs):
    """
    Convert input data to a prepared data bundle for fitting.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        If np.ndarray (1D): treated as HEALPix count map.
        If pd.DataFrame: must contain 'ra' and 'dec' columns (degrees).
    **kwargs
        nside (int): HEALPix NSIDE (default 64).
        nest (bool): HEALPix ordering scheme (default False = RING).
        mask (array-like of bool, optional): footprint mask for HEALPix maps.
    
    Returns
    -------
    dict with keys:
        - data_map: full HEALPix map (npix,)
        - selection: boolean mask of valid pixels used for fitting (npix,)
        - x, y: normalized coordinates [-1,+1] of selected pixels
        - observed: observed counts at selected pixels
        - nside, npix: HEALPix metadata
    """
    nside = int(kwargs.get("nside", 64))
    nest = bool(kwargs.get("nest", False))
    user_mask = kwargs.get("mask", None)
    _validate_nside(nside)
    npix = hp.nside2npix(nside)

    def _coerce_mask(mask, shape):
        if mask is None:
            return np.zeros(shape, dtype=bool)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == ():
            mask_arr = np.full(shape, bool(mask_arr), dtype=bool)
        if mask_arr.shape != shape:
            raise ValueError(f"mask has shape {mask_arr.shape}, expected {shape}.")
        return mask_arr

    if isinstance(data, np.ma.MaskedArray):
        data_map = np.asarray(data.filled(0.0), dtype=float)
        mask = np.ma.getmaskarray(data)
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        data_map = np.asarray(data, dtype=float)
        mask = np.zeros_like(data_map, dtype=bool)
    elif isinstance(data, dict):
        data = pd.DataFrame(data)
    if isinstance(data, pd.DataFrame):
        required_cols = ["ra", "dec"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        ra = data["ra"].to_numpy(dtype=float)
        dec = data["dec"].to_numpy(dtype=float)
        valid = np.isfinite(ra) & np.isfinite(dec)
        pix = hp.ang2pix(nside, ra[valid], dec[valid], lonlat=True, nest=nest)
        data_map = np.zeros(npix, dtype=float)
        np.add.at(data_map, pix, 1.0)
        mask = np.zeros(npix, dtype=bool)
    else:
        if not (isinstance(data, np.ndarray) or isinstance(data, np.ma.MaskedArray)):
            raise ValueError("Data format not recognized. Provide a HEALPix map or a DataFrame with 'ra'/'dec'.")

    if user_mask is not None:
        mask = mask | _coerce_mask(user_mask, data_map.shape)

    if data_map.size != npix:
        raise ValueError(f"Input map size={data_map.size} is inconsistent with nside={nside} (expected npix={npix}).")

    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True, nest=nest)
    x = _normalize_to_minus_one_plus_one(lon)
    y = _normalize_to_minus_one_plus_one(lat)

    selection = (~mask) & np.isfinite(x) & np.isfinite(y)
    fit_idx = np.where(selection)[0] # Indices of pixels used for fitting (unmasked and valid coordinates)
    # We will evaluate the likelihood only on these selected pixels to avoid
    # issues with masked/invalid data. This is better than selection mask, because it gives us the actual indices
    #  to index into the full data_map when computing expected counts, and avoid
    #  some jax issues with boolean indexing.

    return {
        "data_map": data_map,
        "mask": mask,
        "selection": selection,
        "fit_idx": fit_idx,
        "x": x[selection],
        "y": y[selection],
        "observed": np.clip(data_map[selection], 0.0, None),
        "nside": nside,
        "npix": npix,
    }

def _normalize_to_minus_one_plus_one(arr):
    """Normalize array to [-1, +1] range."""
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amax, amin):
        return np.zeros_like(arr)
    return 2.0 * (arr - amin) / (amax - amin) - 1.0

def polyfit2d(x, y, f, deg):
    """
    Fit a 2d polynomial.

    Parameters:
    -----------
    x : array of x values
    y : array of y values
    f : array of function return values
    deg : polynomial degree (length-2 list)

    Returns:
    --------
    c : polynomial coefficients
    """
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)


def process_data_polyfit2d(data, proj, sigma=0.0, percent=None, debug=False, **kwargs):
    """
    Prepare projected coordinates and values for polyfit2d from a HEALPix map.

    Parameters
    ----------
    data : np.ndarray, np.ma.MaskedArray, pd.DataFrame, or dict
        Input sky data accepted by process_data.
    proj : object
        Projection object with ang2xy(...) and optionally get_extent().
    sigma : float, default=0.0
        Gaussian smoothing width in degrees (0 disables smoothing).
    percent : tuple/list or None
        Optional percentile clipping, e.g. (2, 95).
    mask : array-like of bool, optional
        Optional footprint mask for HEALPix maps. True values are excluded from
        the fit. This is useful when a sparse map stores unobserved pixels as 0.

    Returns
    -------
    dict
        Prepared projected data for least-squares polynomial fitting.
    """
    if proj is None or not hasattr(proj, "ang2xy"):
        raise ValueError("polyfit2d method requires `proj` with an `ang2xy` method.")

    prepared = process_data(data, **kwargs)
    nside = prepared["nside"]
    npix = prepared["npix"]
    mask = prepared["mask"]

    data_ma = np.ma.array(prepared["data_map"], mask=mask)

    if percent is not None:
        p_lo, p_hi = percent
        vmin, vmax = np.percentile(data_ma.compressed(), [p_lo, p_hi])
        fill_med = float(np.ma.median(data_ma))
        clipped = np.clip(data_ma.filled(fill_med), vmin, vmax)
        data_ma = np.ma.array(clipped, mask=mask)

    if sigma is not None and float(sigma) > 0:
        fill_med = float(np.ma.median(data_ma))
        smoothed = hp.smoothing(data_ma.filled(fill_med), sigma=np.radians(float(sigma)))
        data_ma = np.ma.array(smoothed, mask=mask)

    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    # Prefer the selection computed by process_data (handles mask and finite lon/lat).
    sel = prepared.get("selection", None)
    if sel is None:
        sel = ~data_ma.mask

    x_sel, y_sel = proj.ang2xy(lon[sel], lat[sel], lonlat=True)
    x_sel = np.asarray(x_sel, dtype=float)
    y_sel = np.asarray(y_sel, dtype=float)
    v_sel = np.asarray(data_ma[sel], dtype=float)

    sel2 = np.isfinite(x_sel) & np.isfinite(y_sel) & np.isfinite(v_sel)
    extent_count = None
    if hasattr(proj, "get_extent"):
        xmin, xmax, ymin, ymax = proj.get_extent()
        # Use inclusive bounds to avoid dropping pixels that lie exactly on the edge.
        extent_mask = (x_sel >= xmin) & (x_sel <= xmax) & (y_sel >= ymin) & (y_sel <= ymax)
        extent_count = int(np.count_nonzero(extent_mask))
        sel2 &= extent_mask

    if debug:
        print("[process_data_polyfit2d debug] sel_count:", int(np.count_nonzero(sel)))
        print("[process_data_polyfit2d debug] proj finite x/y:", int(np.count_nonzero(np.isfinite(x_sel))), int(np.count_nonzero(np.isfinite(y_sel))))
        print("[process_data_polyfit2d debug] v_sel finite:", int(np.count_nonzero(np.isfinite(v_sel))))
        if extent_count is not None:
            print("[process_data_polyfit2d debug] extent_count:", extent_count)
        print("[process_data_polyfit2d debug] sel2_count:", int(np.count_nonzero(sel2)))

    x_fit = x_sel[sel2]
    y_fit = y_sel[sel2]
    v_fit = v_sel[sel2]

    x_all, y_all = proj.ang2xy(lon, lat, lonlat=True)
    x_all = np.asarray(x_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)

    # Debug logging to help trace empty-fit issues
    try:
        LOGGER.debug("process_data_polyfit2d: sel_count=%d, sel2_count=%d, mask_sum=%d", np.count_nonzero(sel), np.count_nonzero(sel2), np.sum(data_ma.mask))
    except Exception:
        pass

    # Normalize projected coordinates for numerical stability of high-order fits.
    finite_x_all = np.isfinite(x_all)
    finite_y_all = np.isfinite(y_all)

    if np.any(finite_x_all):
        x_min, x_max = float(np.nanmin(x_all[finite_x_all])), float(np.nanmax(x_all[finite_x_all]))
    else:
        x_min, x_max = -1.0, 1.0
    if np.any(finite_y_all):
        y_min, y_max = float(np.nanmin(y_all[finite_y_all])), float(np.nanmax(y_all[finite_y_all]))
    else:
        y_min, y_max = -1.0, 1.0

    x_span = x_max - x_min
    y_span = y_max - y_min

    if x_span < 1e-12:
        x_all_norm = np.zeros_like(x_all)
        x_fit_norm = np.zeros_like(x_fit)
    else:
        x_all_norm = 2.0 * (x_all - x_min) / x_span - 1.0
        x_fit_norm = 2.0 * (x_fit - x_min) / x_span - 1.0

    if y_span < 1e-12:
        y_all_norm = np.zeros_like(y_all)
        y_fit_norm = np.zeros_like(y_fit)
    else:
        y_all_norm = 2.0 * (y_all - y_min) / y_span - 1.0
        y_fit_norm = 2.0 * (y_fit - y_min) / y_span - 1.0

    return {
        "nside": nside,
        "npix": npix,
        "mask": np.asarray(data_ma.mask, dtype=bool),
        "data_map": np.asarray(data_ma.filled(0.0), dtype=float),
        "x_fit": x_fit_norm,
        "y_fit": y_fit_norm,
        "v_fit": v_fit,
        "x_all": x_all_norm,
        "y_all": y_all_norm,
        "data_ma": data_ma,
    }

###############