import copy
import functools

import numpy as np
import ugali.isochrone


@functools.lru_cache(maxsize=32)
def _isochrone_factory_cached(params_key):
    """ugali.isochrone.factory(...) memoized on its (hashable) parameters.

    Building a ugali isochrone is expensive out of all proportion to what it
    looks like: `IsochroneModel.create_grid` globs its whole isochrone data
    directory and parses every filename (~11,500 files per construction,
    measured). That cost used to be paid on EVERY call to
    convert_N_SurfaceBrightness -- which convert_SurfaceBrightness_to_N's
    brentq inversion calls ~31 times per stream, for an isochrone that
    depends only on (age, z, survey) and is therefore identical every time.
    Profiling a real training sample put 93% of its wall time here (PLAN.md
    section 6.16), i.e. the pipeline spent almost all of its time rebuilding
    the same object.
    """
    return ugali.isochrone.factory(**dict(params_key))


def _build_isochrone(iso_sub_params):
    """Return the isochrone for `iso_sub_params`, via the cache when possible.

    Callers assign `isochrone.distance_modulus` before use, so handing out
    the cached object itself would let one caller's distance modulus leak
    into every later one. A plain `copy.copy` is NOT enough to prevent that,
    despite looking like it should be: ugali keeps `distance_modulus` as a
    mutable Parameter object inside dicts held on the instance, which a
    shallow copy shares by reference. Two traps found by testing rather than
    reading (each produced a copy that still leaked):

    1. the instance carries BOTH `params` and `_params` in its `__dict__`,
       and `setp` mutates `params` -- copying only `_params` isolates
       nothing;
    2. whether those two are the same dict object is not guaranteed (on
       `Marigo2017` they are distinct), so each is replaced by its own copy
       while any aliasing between them is preserved rather than assumed
       either way.

    Only those small dicts of scalar-valued Parameters are deep-copied; the
    expensive isochrone grid arrays stay shared by reference, so the copy
    stays cheap.

    Falls back to an uncached construction if the parameters aren't hashable
    (e.g. a nested dict), so this can never change what callers are allowed
    to pass.
    """
    try:
        params_key = tuple(sorted(iso_sub_params.items()))
        hash(params_key)
    except TypeError:
        return ugali.isochrone.factory(**iso_sub_params)

    cached = _isochrone_factory_cached(params_key)
    isochrone = copy.copy(cached)
    copies_by_id = {}
    for name in ("params", "_params"):
        if name not in cached.__dict__:
            continue
        source = cached.__dict__[name]
        if id(source) not in copies_by_id:
            copies_by_id[id(source)] = copy.deepcopy(source)
        # object.__setattr__ bypasses ugali's overridden __setattr__, which
        # would route a name that is itself a parameter key into setp().
        object.__setattr__(isochrone, name, copies_by_id[id(source)])
    return isochrone


def convert_N_SurfaceBrightness(N, mag_bounds = (None, 24), surface=None, stream_length=None, stream_width = None, isochrone_config_path =None,
                                band='r',isochrone_params = None, verbose=False):
    if isochrone_params is None:
        if isochrone_config_path is None:
            raise ValueError("Either isochrone_config_path or isochrone_params must be provided.")
        import yaml
        with open(isochrone_config_path, 'r') as f:
            isochrone_params = yaml.safe_load(f)

    iso_sub_params = isochrone_params.get('isochrone', {})

    isochrone = _build_isochrone(iso_sub_params)
    distance_modulus = isochrone_params.get('distance_modulus', {})['center']['value']
    isochrone.distance_modulus = distance_modulus

    # Surface estimation
    if surface is None:
        if stream_length is None or stream_width is None:
            raise ValueError("If surface is not provided, stream_length and stream_width must be provided to estimate surface.")
        sigma_rad = np.deg2rad(stream_width)
        length_rad = np.deg2rad(stream_length)

        surface_sr = length_rad * 2 * np.sin(sigma_rad) # Surface in steradians. Sinus because of sphere integration
        # It's equivalent when the width is small, since sin(sigma) ~ sigma

        surface_deg2 = surface_sr * (180/np.pi)**2 # Convert surface to deg^2

        surface = surface_deg2 * (3600)**2 # Convert surface to arcsec^2
        if verbose:
            print(f"Estimated surface of the stream is {surface_deg2:.2f}deg^2 (length={stream_length}, width={stream_width})")
    

    N_in_surface = int(N*0.68) # 68% of stars are within 1 sigma of the gaussian profile of the stream
    tot_magnitude = convert_N_to_luminosity(N_in_surface, isochrone=isochrone, mag_bounds=mag_bounds,verbose=verbose, band=band)
    surface_brightness = tot_magnitude + 2.5 * np.log10(surface) # Surface brightness in mag/arcdeg^2

    if verbose:
        print(f"Estimated surface brightness is {surface_brightness:.2f} mag/arcsec^2 for N={N}, N_in_surface={N_in_surface}, tot_magnitude={tot_magnitude:.2f}, surface={surface:.2f} arcsec^2")

    return surface_brightness


def convert_N_to_luminosity(N, isochrone, mag_bounds= (None,24),verbose=False, band= 'r'):
    mass_init, mass_pdf, mass_act, mag_g, mag_r = isochrone.sample(mass_steps=10000)
    mag_g, mag_r = mag_g + isochrone.distance_modulus, mag_r + isochrone.distance_modulus
    if band == 'g':
        mag1 = mag_g
    elif band == 'r':
        mag1 = mag_r
    else:
        raise ValueError(f"Band '{band}' not recognized. Use 'g' or 'r'.")
    
    mask = np.ones_like(mag1, dtype=bool)
    if mag_bounds[0] is not None:
        mask &= (mag1 > mag_bounds[0])
    if mag_bounds[1] is not None:
        mask &= (mag1 < mag_bounds[1])
    mass_pdf_norm = mass_pdf / np.sum(mass_pdf) # Normalize the mass pdf to get a proper probability distribution function
    number_of_stars_per_mass_bin = mass_pdf_norm * N

    #number_of_stars_per_mass_bin = mass_pdf * N # Number of stars corresponding to each mass step, given the pdf and total N
    number_of_stars_per_mass_bin_within_mag_bounds = number_of_stars_per_mass_bin[mask] # Number of stars corresponding to each mass step, but only for the mass steps that have mag within bounds
    
    # Select only the magnitudes corresponding to the mass steps that have mag within bounds
    mag_sel = mag1[mask]
    lum = magToFlux(mag_sel) # Luminosity corresponding to the magnitude of each mass step 

    L_tot = np.sum(number_of_stars_per_mass_bin_within_mag_bounds * lum)
    M_tot = fluxToMag(L_tot)
    
    return M_tot


def magToFlux(mag):
    """
    Convert from AB magnitude to flux.

    Parameters
    ----------
    mag : float or np.ndarray
        AB magnitude(s).

    Returns
    -------
    float or np.ndarray
        Flux in Janskys (Jy).
    """
    return 3631.0 * 10 ** (-0.4 * mag)

def fluxToMag(flux):
    """
    Convert from flux to AB magnitude.

    Parameters
    ----------
    flux : float or np.ndarray
        Flux in Janskys (Jy).

    Returns
    -------
    float or np.ndarray
        AB magnitude(s).
    """
    return -2.5 * np.log10(flux / 3631.0)


def convert_SurfaceBrightness_to_N(target_surface_brightness, mag_bounds=(None, 24), surface=None,
                                    stream_length=None, stream_width=None, isochrone_config_path=None,
                                    band='r', isochrone_params=None, n_bracket=(10, 1e8), verbose=False):
    """
    Numeric inverse of convert_N_SurfaceBrightness: the number of stars N whose
    stream (with the given isochrone/surface parameters) has the requested
    surface brightness.

    convert_N_SurfaceBrightness(N, ...) is monotonic in N (more stars -> more
    total flux -> brighter/smaller surface brightness), so this brackets and
    solves for N in log10(N) space with scipy.optimize.brentq, calling
    convert_N_SurfaceBrightness itself (same isochrone-loading path, mag_bounds,
    and surface as the forward function -- nothing here is duplicated).

    Parameters
    ----------
    target_surface_brightness : float
        Desired surface brightness, mag/arcsec^2 (same convention as
        convert_N_SurfaceBrightness's return value).
    mag_bounds, surface, stream_length, stream_width, isochrone_config_path, band, isochrone_params, verbose : see convert_N_SurfaceBrightness.
    n_bracket : tuple (N_min, N_max)
        Bracket searched in log10(N) space. Widen this if brentq raises a
        sign-mismatch error (the target surface brightness falls outside what
        this bracket can produce).

    Returns
    -------
    float
        N (number of stars) giving `target_surface_brightness`.
    """
    from scipy.optimize import brentq

    def _residual(log10_N):
        N = 10 ** log10_N
        sb = convert_N_SurfaceBrightness(
            N, mag_bounds=mag_bounds, surface=surface, stream_length=stream_length,
            stream_width=stream_width, isochrone_config_path=isochrone_config_path,
            band=band, isochrone_params=isochrone_params, verbose=False,
        )
        return sb - target_surface_brightness

    log10_N = brentq(_residual, np.log10(n_bracket[0]), np.log10(n_bracket[1]))
    N = 10 ** log10_N

    if verbose:
        print(f"N={N:.1f} gives surface brightness {target_surface_brightness:.2f} mag/arcsec^2")

    return N


def _load_isochrone(isochrone_config_path=None, isochrone_params=None):
    """Build the ugali isochrone described by isochrone_config_path/isochrone_params
    (same 'isochrone' sub-section shape convert_N_SurfaceBrightness uses)."""
    if isochrone_params is None:
        if isochrone_config_path is None:
            raise ValueError("Either isochrone_config_path or isochrone_params must be provided.")
        import yaml
        with open(isochrone_config_path, 'r') as f:
            isochrone_params = yaml.safe_load(f)

    iso_sub_params = isochrone_params.get('isochrone', {})
    return _build_isochrone(iso_sub_params)


def convert_N_to_Mass(N, isochrone_config_path=None, isochrone_params=None, mass_min=0.1):
    """
    Total stellar mass (Msun) of N stars drawn from the isochrone's IMF.

    mass = N * isochrone.stellar_mass(mass_min=mass_min): stellar_mass() is
    already the IMF-weighted mean initial mass per star (Msun) -- the same
    N <-> mass relationship ugali's own IsochroneModel.simulate() uses
    internally (``richness = stellar_mass / self.stellar_mass()``), not a new
    convention invented here. Unlike convert_N_SurfaceBrightness, this needs
    no distance modulus or magnitude bounds -- mass counts all stars, not
    just the observable ones.

    Parameters
    ----------
    N : float
        Number of stars.
    isochrone_config_path, isochrone_params : see convert_N_SurfaceBrightness.
    mass_min : float
        Minimum mass to integrate the IMF over (Msun). Default 0.1, matching
        ugali's IsochroneModel.stellar_mass()/sample() default.

    Returns
    -------
    float
        Total stellar mass (Msun).
    """
    isochrone = _load_isochrone(isochrone_config_path, isochrone_params)
    return N * isochrone.stellar_mass(mass_min=mass_min)


def convert_Mass_to_N(mass, isochrone_config_path=None, isochrone_params=None, mass_min=0.1):
    """
    Numeric inverse of convert_N_to_Mass: N = mass / mean_stellar_mass_per_star.

    Parameters
    ----------
    mass : float
        Total stellar mass (Msun).
    isochrone_config_path, isochrone_params, mass_min : see convert_N_to_Mass.

    Returns
    -------
    float
        N (number of stars) giving this total mass.
    """
    isochrone = _load_isochrone(isochrone_config_path, isochrone_params)
    return mass / isochrone.stellar_mass(mass_min=mass_min)