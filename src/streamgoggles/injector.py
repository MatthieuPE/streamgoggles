"""Single-stream and population-stream injection into background.

Pipeline for one stream:
1. Realize stream (phi1, phi2, dist, true mags) via StreamSource.
2. Place in footprint + sample window (with ≥5° overlap constraint).
3. Inject through survey selection (streamobs.StreamInjector) → observed mags + flags.
4. Apply background cuts + clipping (same as background, so selection is consistent).
5. For each trial distance: select + make_raw_map.
6. Combine with cached background, optionally finalize, crop window.
7. Rasterize stream labels.
8. Return Sample.

Pure-background samples (no stream) skip steps 1, 3–5, 7; window placed randomly.

Population injection (stub, decision 19): sum raw stream maps, union labels.
Designed to compose single-stream path without rewriting.

Rationale: Isolate injection logic from background/window/filter. Enable testing
injection independently. Keep code simple: one stream at a time, then compose.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StreamInjector:
    """Inject stream(s) into background, apply survey effects, build sample.
    
    Attributes:
        background: Background instance (catalog + cached maps).
        matched_filter: MatchedFilter instance.
        stream_source: StreamSource instance.
        cuts: List of Cut rules (applied to stream as well as background).
        clipping: Magnitude clipping dict.
        pix: PixelizationSpec.
    
    Rationale: Encapsulate injection logic; enable reuse with different
    backgrounds/filters/sources by swapping attributes.
    """
    
    def __init__(
        self,
        background: "Background",
        matched_filter: "MatchedFilter",
        stream_source: "StreamSource",
        cuts: list["Cut"],
        clipping: dict | None,
        pix: "PixelizationSpec"
    ):
        """Initialize injector.
        
        Parameters:
            background: Background instance.
            matched_filter: MatchedFilter instance.
            stream_source: StreamSource instance.
            cuts: List of Cut rules.
            clipping: Magnitude clipping dict.
            pix: PixelizationSpec.
        """
        self.background = background
        self.matched_filter = matched_filter
        self.stream_source = stream_source
        self.cuts = cuts or []
        self.clipping = clipping or {}
        self.pix = pix
    
    def inject_single_stream(
        self,
        params: dict,
        window: "Window",
        rng: np.random.Generator
    ) -> "Sample":
        """Inject one stream into background; return labeled sample.
        
        Pipeline:
        1. Realize stream via stream_source (phi1, phi2, dist, true mags).
        2. Place stream at random position/orientation in footprint.
        3. Sample window (rejection-sample for ≥5° overlap).
        4. Inject through streamobs.StreamInjector (survey selection + errors).
        5. Apply background cuts + clipping.
        6. For each distance: select + make_raw_map.
        7. Combine with cached background, finalize if enabled, crop window.
        8. Rasterize stream labels.
        9. Return Sample.
        
        Parameters:
            params: Stream parameter dict
                (morphology, width, length, distance_modulus, age, z, nstars, etc.).
            window: Window to crop to (center, rotation, size).
            rng: Random number generator.
        
        Returns:
            Sample(map_stack, label_stack, valid_mask, params, metadata).
        
        Raises:
            RuntimeError if window sampling fails (e.g., stream conflicts with footprint).
                (Note: This should rarely happen if the stream footprint is well-defined.)
        """
        raise NotImplementedError
    
    def inject_population(
        self,
        params_list: list[dict],
        window: "Window",
        rng: np.random.Generator
    ) -> "Sample":
        """Inject multiple streams into one window (stub, decision 19).
        
        Combines single-stream injection results:
        - Sum raw stream maps (additive; valid because unweighted).
        - Union labels (both sets of rasterized members).
        - Return combined Sample.
        
        Parameters:
            params_list: List of parameter dicts (one per stream).
            window: Single window for all streams.
            rng: Random number generator.
        
        Returns:
            Sample with summed maps and unioned labels.
        
        Raises:
            NotImplementedError (Stage 1 stub; implement in Stage 2b).
        """
        raise NotImplementedError


def inject_background_only(
    background: "Background",
    window: "Window",
    pix: "PixelizationSpec"
) -> "Sample":
    """Generate a pure-background (no stream) sample.
    
    Parameters:
        background: Background instance.
        window: Window to crop to.
        pix: PixelizationSpec.
    
    Returns:
        Sample with map_stack from background, label_stack all zeros.
    
    Rationale: Teaches network not to hallucinate streams in every field.
    Used when background_fraction > 0 in config.
    """
    raise NotImplementedError
