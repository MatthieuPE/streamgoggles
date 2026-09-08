"""Pluggable stream source implementations (streamobs, external sims, etc).

Rationale: Abstract "where stream stars come from" so the injector doesn't care 
whether it's streamobs, pre-realized external sims, or future sources. Stream stars
are realized in stream frame (phi1, phi2, dist) with TRUE magnitudes and converted
to sky coordinates on-the-fly.
"""

import logging
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StreamSource(Protocol):
    """Protocol for generating stream star catalogs.
    
    All implementations realize stream in stream frame (phi1, phi2, dist),
    returning TRUE (unreddened) magnitudes and an is_stream flag.
    
    Rationale: Protocol enables pluggable sources; downstream injection
    applies survey selection and coordinate transformation to all sources
    identically.
    """
    
    def realize(
        self,
        params: dict,
        rng: np.random.Generator
    ) -> pd.DataFrame:
        """Realize one stream sample.
        
        Parameters:
            params: Stream parameter dict with keys:
                morphology: "uniform" or "spline"
                width: stream width (degrees)
                length: stream length (degrees)
                distance_modulus: true distance modulus
                age: age in Gyr
                z: metallicity Z (mass fraction)
                nstars: number of stream stars to generate
                (other params as needed per morphology / source)
            
            rng: np.random.Generator instance for reproducibility.
        
        Returns:
            DataFrame with columns:
                phi1, phi2: stream frame coordinates (degrees)
                dist: distance (kpc or as needed)
                lsst_*_true: true magnitudes in streamobs convention (before survey errors)
                is_stream: bool, always True
                (any other columns the source wishes to include)
        
        Raises:
            ValueError if params are invalid or inconsistent.
        """


class StreamObsSource(StreamSource):
    """Generate stream via streamobs.model.StreamModel.
    
    Wraps streamobs's stream generation, using the isochrone and survey
    parameters passed through params.
    
    Rationale: streamobs is the canonical stream generation library.
    Wrapping it here allows reuse without importing streamobs everywhere,
    and handles conversion from user-friendly richness specs (surface
    brightness, mass, nstars) to streamobs's nstars parameter.
    """
    
    def __init__(self):
        """Initialize streamobs source."""
        pass
    
    def realize(
        self,
        params: dict,
        rng: np.random.Generator
    ) -> pd.DataFrame:
        """Realize stream via streamobs.
        
        Parameters:
            params: dict with keys matching StreamObsSource's expectations
                (morphology, width, length, distance_modulus, age, z, nstars, etc.).
        
        Returns:
            DataFrame in stream frame with true magnitudes (unreddened).
        
        Note: nstars must be precomputed from richness (surface_brightness/mass)
        using the user's conversion functions in inject_utils.py, NOT computed here.
        """
        raise NotImplementedError


class ExternalSimSource(StreamSource):
    """Load pre-realized stream catalogs from data/external_sims/stream_{id}/.
    
    Minimal required columns: phi1, phi2.
    Optional columns: true magnitudes per band (streamobs convention),
        age, z (metallicity Z).
    
    If magnitudes/age/z are missing, fallback: call streamobs isochrone
    sampling using params['age'], params['z'].
    
    Rationale: Enables use of external stream simulations (e.g., N-body outputs)
    without rewriting them to streamobs format. Designed to slot in seamlessly.
    
    Stage 1: Stub (raise NotImplementedError).
    Stage 2+: Full implementation after StreamObsSource is validated.
    """
    
    def __init__(self):
        """Initialize external sim source."""
        pass
    
    def realize(
        self,
        params: dict,
        rng: np.random.Generator
    ) -> pd.DataFrame:
        """Load pre-realized stream from disk or fallback to streamobs sampling.
        
        Parameters:
            params: dict with:
                stream_id: "stream_NNN" or similar
                age (optional): Gyr (used if magnitudes missing in realization)
                z (optional): metallicity Z (used if missing in realization)
        
        Returns:
            DataFrame with phi1, phi2, and (either loaded or sampled) magnitudes.
        
        Raises:
            FileNotFoundError if stream_id directory not found.
            NotImplementedError (Stage 1 stub).
        """
        raise NotImplementedError
