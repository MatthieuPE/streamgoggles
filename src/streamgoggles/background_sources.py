"""Pluggable background sources (real DP2, synthetic LSST yr1, etc).

Three implementations (decision 23):
1. DataFileBackgroundSource: real DP2 parquet (+ dust correction).
2. StreamObsLightBackgroundSource: streamobs's fast synthetic background.
3. StreamObsCatalogueBackgroundSource: streamobs's fuller synthetic catalog.

All three produce the same downstream pipeline input (streamobs column convention),
so cuts/clipping/select/caching are identical regardless of source.

Rationale: Enables starting work against LSST yr1 forecast (no real data) before
switching to real DP2 once released. Same code path throughout.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StudyRegion:
    """Spatial region for synthetic background generation (decision 24).
    
    Attributes:
        center_ra: Region center RA (degrees).
        center_dec: Region center Dec (degrees).
        width_deg: Region width in degrees (default 70).
        height_deg: Region height in degrees (default 20).
    
    Rationale: Restricts synthetic background generation to a region much smaller
    than LSST's enormous footprint, saving compute. Only used by light/catalogue
    sources; data_file sources ignore it (real catalog is used in full).
    """
    center_ra: float
    center_dec: float
    width_deg: float = 70.0
    height_deg: float = 20.0
    
    def is_inside(self, ra: pd.Series | list, dec: pd.Series | list) -> pd.Series | list:
        """Return boolean mask: stars inside this region.
        
        Parameters:
            ra, dec: RA and Dec arrays/series (degrees).
        
        Returns:
            Boolean mask (same type/length as inputs).
        
        Rationale: Mask can then be used to filter catalog: 
            df = df[region.is_inside(df['ra'], df['dec'])].
        """
        raise NotImplementedError


class BackgroundSource(Protocol):
    """Protocol for loading background catalogs from various sources.
    
    All implementations return a catalog in streamobs column convention,
    enabling identical downstream processing.
    
    Rationale: Pluggable sources decouple "where background comes from" 
    from "how to process it". Enables switching between real/synthetic 
    data with a config change only.
    """
    
    def load(
        self,
        survey: str,
        release: str,
        region: StudyRegion | None,
        cfg: dict
    ) -> pd.DataFrame:
        """Load background star catalog.
        
        Parameters:
            survey: Survey name (e.g., "lsst").
            release: Release name (e.g., "dp2", "yr1").
            region: StudyRegion (used only by light/catalogue sources; ignored by data_file).
            cfg: Source-specific config dict (e.g., {path: ...} for data_file).
        
        Returns:
            DataFrame with columns in streamobs convention
            (e.g., 'lsst_g_obs', 'lsst_r_obs', 'lsst_g_err', 'lsst_r_err', etc.)
            and standard coordinate columns ('ra', 'dec').
        
        Rationale: Standardized output format enables downstream code
        (cuts, clipping, select) to be identical regardless of source.
        """


class DataFileBackgroundSource(BackgroundSource):
    """Load real DP2 from parquet; apply dust correction; ignore region.
    
    Rationale: Real data sources don't need region restriction (the catalog
    is already a bounded skim of the full survey). Dust correction is mandatory
    (decision 25) to match observed magnitudes to isochrones.
    
    Attributes:
        path_key: Key in cfg dict pointing to file path (default "path").
    """
    
    def __init__(self, path_key: str = "path"):
        """Initialize source.
        
        Parameters:
            path_key: Key in cfg dict for file path (default "path").
        """
        self.path_key = path_key
    
    def load(
        self,
        survey: str,
        release: str,
        region: StudyRegion | None,
        cfg: dict
    ) -> pd.DataFrame:
        """Load DP2 parquet, harmonize columns, apply dust correction.
        
        Ignores region (region is not passed to downstream code for data_file).
        
        Parameters:
            survey, release: Documentation/provenance (the real catalog is what it is).
            region: Ignored.
            cfg: dict with key self.path_key -> file path, e.g., cfg['path'] = 
                'data/background/dp2_star_gmax_27_skim.parquet'.
        
        Returns:
            DataFrame in streamobs convention, dust-corrected.
        
        Raises:
            FileNotFoundError if cfg[self.path_key] doesn't exist.
        """
        raise NotImplementedError


class StreamObsLightBackgroundSource(BackgroundSource):
    """Wrap streamobs's fast/light synthetic background generation.
    
    Rationale: Fast, good for early testing. Region-restricted at generation time
    to avoid generating unnecessary far-field stars.
    """
    
    def load(
        self,
        survey: str,
        release: str,
        region: StudyRegion | None,
        cfg: dict
    ) -> pd.DataFrame:
        """Generate synthetic background via streamobs.
        
        Parameters:
            survey: Survey name (e.g., "lsst").
            release: Release name (e.g., "yr1").
            region: StudyRegion for generation (if None, uses default study region).
            cfg: Source-specific kwargs for streamobs generation (e.g., {}).
        
        Returns:
            DataFrame in streamobs convention, region-restricted.
        
        Raises:
            ValueError if release is not supported by streamobs.
        """
        raise NotImplementedError


class StreamObsCatalogueBackgroundSource(BackgroundSource):
    """Wrap streamobs's fuller realistic synthetic catalog generation.
    
    Rationale: More realistic than light, but slower. Region-restricted at 
    generation time.
    """
    
    def load(
        self,
        survey: str,
        release: str,
        region: StudyRegion | None,
        cfg: dict
    ) -> pd.DataFrame:
        """Generate synthetic background via streamobs (catalogue backend).
        
        Parameters:
            survey: Survey name (e.g., "lsst").
            release: Release name (e.g., "yr1").
            region: StudyRegion for generation (if None, uses default study region).
            cfg: Source-specific kwargs for streamobs generation.
        
        Returns:
            DataFrame in streamobs convention, region-restricted.
        
        Raises:
            ValueError if release is not supported by streamobs.
        """
        raise NotImplementedError


def harmonize_columns(
    df: pd.DataFrame,
    source: str = "dp2"
) -> pd.DataFrame:
    """Rename/derive DP2 columns to streamobs convention.
    
    Parameters:
        df: Input DataFrame with DP2 column names.
        source: Source type ("dp2" only for now).
    
    Returns:
        DataFrame with streamobs-convention columns
        (e.g., 'lsst_g_obs', 'lsst_r_obs', etc.).
    
    Rationale: Enables the same downstream select() function to work on
    both background and stream stars (both must be in same convention).
    
    Raises:
        ValueError if source is not supported.
    """
    raise NotImplementedError
