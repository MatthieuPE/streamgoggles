"""Footprints, and the catalogued objects that must not be called streams."""

import healpy as hp
import numpy as np
import pytest

from streamgoggles.objects_overlap import (
    get_dwarf,
    get_dwarf_within_footprint,
    get_footprint,
    get_GC,
    get_GC_within_footprint,
    mask_objects,
)


def test_no_footprint_is_the_whole_sky():
    mask, pixels, npix = get_footprint(nside=64)
    assert npix == hp.nside2npix(64)
    assert mask.all() and pixels.size == npix


def test_named_footprint_is_the_des_survey():
    mask, pixels, _ = get_footprint("des_yr6")
    area = mask.sum() * hp.nside2pixarea(512, degrees=True)
    assert 4000 < area < 6000, f"DES Y6 covers about 5000 deg^2, got {area:.0f}"
    assert pixels.size == mask.sum()


def test_footprint_survives_a_change_of_resolution():
    """Regridding must not eat the footprint: a partly covered coarse pixel
    counts as covered, so the area is stable rather than shrinking."""
    fine = get_footprint("des_yr6", nside=512)[0]
    coarse = get_footprint("des_yr6", nside=256)[0]
    fine_area = fine.sum() * hp.nside2pixarea(512, degrees=True)
    coarse_area = coarse.sum() * hp.nside2pixarea(256, degrees=True)
    assert coarse_area == pytest.approx(fine_area, rel=0.05)


def test_footprint_accepts_a_mask_and_a_pixel_list():
    pixels = np.array([1, 5, 9], dtype=np.int64)
    from_list = get_footprint(pixels, nside=32)[0]
    from_mask = get_footprint(from_list, nside=32)[0]
    assert from_list.sum() == 3
    assert np.array_equal(from_list, from_mask)


def test_footprint_rejects_a_mask_of_the_wrong_size():
    with pytest.raises(ValueError, match="pixels"):
        get_footprint(np.ones(12, dtype=bool), nside=512)


def test_unknown_footprint_name_says_what_is_available():
    with pytest.raises(FileNotFoundError, match="des_yr6"):
        get_footprint("no_such_survey")


def test_catalogues_load_from_the_package():
    """No network: both catalogues ship with the package."""
    for catalogue in (get_GC(), get_dwarf()):
        assert len(catalogue) > 100
        assert {"ra", "dec", "rhalf"} <= set(catalogue.colnames)


def test_des_footprint_holds_some_objects_but_not_most_of_them():
    gc, gc_in = get_GC_within_footprint(footprint="des_yr6")
    dwarf, dwarf_in = get_dwarf_within_footprint(footprint="des_yr6")
    # DES covers about an eighth of the sky, well away from the galactic plane.
    assert 0 < gc_in.sum() < len(gc) / 2
    assert 0 < dwarf_in.sum() < len(dwarf) / 2


def test_masking_objects_scales_with_their_size():
    dwarf, inside = get_dwarf_within_footprint(footprint="des_yr6")
    small, radii_small = mask_objects(dwarf, inside, nside=256, radius_factor=1.0)
    large, radii_large = mask_objects(dwarf, inside, nside=256, radius_factor=5.0)
    assert (radii_large >= radii_small).all()
    assert large.sum() > small.sum()
    # Everything the small mask covers, the large one covers too.
    assert (small & ~large).sum() == 0


def test_masking_gives_an_object_with_no_size_the_floor_radius():
    dwarf, inside = get_dwarf_within_footprint(footprint="des_yr6")
    _, radii = mask_objects(dwarf, inside, nside=256, min_radius_deg=0.2)
    assert radii.min() >= 0.2


# ---------------------------------------------------------------------------
# Known streams
# ---------------------------------------------------------------------------


def test_stream_widths_match_the_ones_streams_are_injected_with():
    """A stream should be masked at the width it is simulated with."""
    import importlib.util
    from pathlib import Path

    from streamgoggles.objects_overlap import DES2018_STREAM_WIDTHS

    path = Path(__file__).parents[1] / "scripts/experiments/stream_parameters/run.py"
    spec = importlib.util.spec_from_file_location("sp_run", path)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    injected = {name: width for name, (width, *_) in run.DES_STREAMS.items()}
    assert injected == DES2018_STREAM_WIDTHS


def test_a_broad_structure_is_masked_to_its_width_not_a_multiple():
    from streamgoggles.objects_overlap import mask_radius

    assert mask_radius(0.5) == pytest.approx(1.5)  # cold stream: 3 sigma
    assert mask_radius(6.0) == pytest.approx(6.0)  # Sagittarius: its extent


def test_stream_names_match_exactly_and_references_can_be_dropped():
    from streamgoggles.objects_overlap import stream_tracks

    jhelum = stream_tracks("Jhelum")
    assert set(jhelum) == {"Jhelum.ibata2021", "Jhelum.ibata2024"}
    # Bonaca's two components are separate names, not caught by "Jhelum".
    assert not any("Jhelum-a" in ref or "Jhelum-b" in ref for ref in jhelum)
    trimmed = stream_tracks("Jhelum", exclude={"Jhelum.ibata2024"})
    assert set(trimmed) == {"Jhelum.ibata2021"}


def test_every_des2018_stream_has_a_track():
    from streamgoggles.objects_overlap import DES2018_STREAM_WIDTHS, stream_tracks

    missing = [name for name in DES2018_STREAM_WIDTHS if not stream_tracks(name)]
    assert not missing, f"no galstreams track for {missing}"


def test_dropping_the_long_jhelum_track_shrinks_its_mask():
    from streamgoggles.objects_overlap import mask_streams

    _, full = mask_streams({"Jhelum": 1.16}, nside=128)
    _, trimmed = mask_streams({"Jhelum": 1.16}, nside=128, exclude={"Jhelum.ibata2024"})
    assert trimmed["Jhelum"].size < 0.6 * full["Jhelum"].size


def test_background_mask_is_the_footprint_minus_both_masks():
    from streamgoggles.objects_overlap import build_background_mask

    m = build_background_mask(nside=128)
    assert not (m["usable"] & ~m["footprint"]).any()
    assert not (m["usable"] & m["streams"]).any()
    assert not (m["usable"] & m["objects"]).any()
    assert np.array_equal(m["usable"], m["footprint"] & ~m["streams"] & ~m["objects"])
    # Streams cost a real share of DES, objects a small one.
    share = (m["streams"] & m["footprint"]).sum() / m["footprint"].sum()
    assert 0.2 < share < 0.45


def test_the_background_mask_is_available_by_name():
    """Training and evaluation read the same sky, by name."""
    usable, _, _ = get_footprint("des_yr6_background")
    footprint, _, _ = get_footprint("des_yr6")
    assert usable.sum() < footprint.sum()
    assert not (usable & ~footprint).any()
    kept = usable.sum() / footprint.sum()
    assert 0.55 < kept < 0.8, f"about 68% of DES is usable, got {100 * kept:.1f}%"
