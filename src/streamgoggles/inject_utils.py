import numpy as np
import ugali.isochrone

def convert_N_SurfaceBrightness(N, mag_bounds = (None, 24), surface=None, stream_length=None, stream_width = None, isochrone_config_path =None,
                                band='r',isochrone_params = None, verbose=False):
    if isochrone_params is None:
        if isochrone_config_path is None:
            raise ValueError("Either isochrone_config_path or isochrone_params must be provided.")
        import yaml
        with open(isochrone_config_path, 'r') as f:
            isochrone_params = yaml.safe_load(f)

    iso_sub_params = isochrone_params.get('isochrone', {})

    isochrone = ugali.isochrone.factory(**iso_sub_params)
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
    return ugali.isochrone.factory(**iso_sub_params)


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