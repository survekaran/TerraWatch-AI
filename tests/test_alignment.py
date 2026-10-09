import hashlib
import numpy as np
import pytest
from scipy.ndimage import shift

from geoai04.alignment import (
    AlignmentConfig,
    AlignmentError,
    apply_alignment_correction,
    estimate_alignment,
)


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_sign_convention_and_shift_recovery(tmp_path, tif):
    """Test that sub-pixel and integer shifts are recovered with the documented sign convention.

    SIGN CONVENTION:
    If AFTER features are translated by (+dr, +dc) in pixel space relative to BEFORE,
    the returned correction vector is (-dr, -dc), such that:
    scipy.ndimage.shift(after, (shift_row_px, shift_col_px)) aligns AFTER with BEFORE.
    """
    rng = np.random.default_rng(42)
    # Textured base image
    base = rng.integers(30, 220, (3, 256, 256), dtype="uint8")

    test_cases = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 2.0),
        (-2.0, 3.0),
        (0.5, -0.5),
        (5.0, -5.0),
    ]

    cfg = AlignmentConfig(n_windows=9, window_size_px=128, upsample_factor=50, texture_threshold=0.01)

    for dr, dc in test_cases:
        # shifted_after has features moved by (+dr, +dc)
        shifted_after = np.zeros_like(base)
        for b in range(3):
            shifted_after[b] = np.clip(
                shift(base[b].astype(float), (dr, dc), order=1, mode="nearest"),
                0, 255
            ).astype("uint8")

        p_before = tif(tmp_path / f"before_{dr}_{dc}.tif", base)
        p_after = tif(tmp_path / f"after_{dr}_{dc}.tif", shifted_after)

        res = estimate_alignment(p_before, p_after, cfg=cfg, gsd_x_m=0.5, gsd_y_m=0.5)

        # Expected correction shift is (-dr, -dc)
        expected_row = -dr
        expected_col = -dc

        assert res.shift_row_px == pytest.approx(expected_row, abs=0.15), (
            f"Failed for shift ({dr}, {dc}): got row {res.shift_row_px}, expected {expected_row}"
        )
        assert res.shift_col_px == pytest.approx(expected_col, abs=0.15), (
            f"Failed for shift ({dr}, {dc}): got col {res.shift_col_px}, expected {expected_col}"
        )
        assert res.magnitude_px == pytest.approx(np.hypot(expected_row, expected_col), abs=0.2)
        assert res.shift_m == pytest.approx(res.magnitude_px * 0.5, abs=0.1)


def test_low_texture_gives_unknown(tmp_path, tif):
    """Low-texture / flat images cannot be reliably aligned and must return status 'unknown'."""
    flat = np.full((3, 256, 256), 120, dtype="uint8")
    p_b = tif(tmp_path / "flat_b.tif", flat)
    p_a = tif(tmp_path / "flat_a.tif", flat)

    cfg = AlignmentConfig(texture_threshold=0.05)
    res = estimate_alignment(p_b, p_a, cfg=cfg)

    assert res.status == "unknown"
    assert res.shift_row_px is None
    assert res.shift_col_px is None
    assert res.magnitude_px is None
    assert any("Too few textured" in r for r in res.reasons)


def test_localised_change_does_not_wreck_estimate(tmp_path, tif):
    """A localised change in a small fraction of the image should not distort the median shift."""
    rng = np.random.default_rng(101)
    base = rng.integers(20, 230, (3, 256, 256), dtype="uint8")

    # Shift after by 1.0 px in col (features moved col +1 -> correction is (0, -1))
    shifted = np.zeros_like(base)
    for b in range(3):
        shifted[b] = shift(base[b].astype(float), (0.0, 1.0), order=1, mode="nearest").astype("uint8")

    # Introduce a big real change in one quadrant (row 20:60, col 20:60)
    shifted[:, 20:60, 20:60] = 250

    p_b = tif(tmp_path / "chg_b.tif", base)
    p_a = tif(tmp_path / "chg_a.tif", shifted)

    cfg = AlignmentConfig(n_windows=9, window_size_px=100, upsample_factor=20)
    res = estimate_alignment(p_b, p_a, cfg=cfg)

    assert res.status in {"pass", "warn"}
    assert res.shift_row_px == pytest.approx(0.0, abs=0.2)
    assert res.shift_col_px == pytest.approx(-1.0, abs=0.2)


def test_auto_correct_preserves_original_sha256_and_reduces_residual(tmp_path, tif):
    """Auto-correction writes a corrected image, leaves originals unchanged, and reduces shift."""
    rng = np.random.default_rng(202)
    base = rng.integers(20, 230, (3, 256, 256), dtype="uint8")

    dr, dc = 3.0, -2.0
    shifted = np.zeros_like(base)
    for b in range(3):
        shifted[b] = shift(base[b].astype(float), (dr, dc), order=1, mode="nearest").astype("uint8")

    p_b = tif(tmp_path / "orig_b.tif", base)
    p_a = tif(tmp_path / "orig_a.tif", shifted)

    sha_before = _hash_file(p_b)
    sha_after = _hash_file(p_a)

    cfg = AlignmentConfig(n_windows=9, window_size_px=128, upsample_factor=20)
    res = estimate_alignment(p_b, p_a, cfg=cfg)
    assert res.status == "fail"  # 3.6 px > 2.0 px fail threshold

    corrected_path = tmp_path / "after_corrected.tif"
    apply_alignment_correction(
        p_a, corrected_path, res.shift_row_px, res.shift_col_px, interpolation="linear"
    )

    # Originals must remain byte-identical
    assert _hash_file(p_b) == sha_before
    assert _hash_file(p_a) == sha_after

    # Re-measure residual shift
    residual = estimate_alignment(p_b, corrected_path, cfg=cfg)
    assert residual.status == "pass"
    assert residual.magnitude_px < 0.4


def test_shifted_pair_raises_alignment_error_and_writes_provenance(tmp_path):
    """A shifted pair (> fail_px) must raise AlignmentError by default and write failed provenance."""
    import json
    from geoai04.config import load_config
    from geoai04.pipeline import run_pipeline
    from geoai04.synthetic import generate_synthetic_pair

    syn_dir = tmp_path / "syn_shifted"
    generate_synthetic_pair(syn_dir, shift_px=(3.0, 0.0))

    out_dir = tmp_path / "out_align"
    with pytest.raises(AlignmentError) as exc_info:
        run_pipeline(
            syn_dir / "before.tif",
            syn_dir / "after.tif",
            load_config(),
            out_root=out_dir,
            run_name="fail_test",
        )

    r_dir = exc_info.value.run_dir
    assert (r_dir / "provenance.json").is_file()
    assert (r_dir / "validation_report.json").is_file()
    assert (r_dir / "run.log").is_file()

    prov = json.loads((r_dir / "provenance.json").read_text())
    assert prov["status"] == "failed_alignment"
    assert prov["alignment"]["status"] == "fail"


def test_shifted_pair_allow_misaligned_completes_with_review_override(tmp_path):
    """allow_misaligned=True allows run to finish, recording override and marking polygons Review."""
    import json
    from geoai04.config import load_config
    from geoai04.pipeline import run_pipeline
    from geoai04.synthetic import generate_synthetic_pair

    syn_dir = tmp_path / "syn_override"
    generate_synthetic_pair(syn_dir, shift_px=(3.0, 0.0))

    out_dir = tmp_path / "out_override"
    res = run_pipeline(
        syn_dir / "before.tif",
        syn_dir / "after.tif",
        load_config(),
        out_root=out_dir,
        run_name="override_test",
        allow_misaligned=True,
    )

    assert res.status == "completed"
    assert any("Misaligned pair override" in w for w in res.warnings)
    assert (res.polygons["severity_tier"] == "Review").all()
    assert res.polygons["quality_flag"].str.contains("alignment_override").all()

    prov = json.loads((res.run_dir / "provenance.json").read_text())
    assert prov["alignment"]["allow_misaligned"] is True


def test_shifted_pair_auto_correct_reduces_shift_and_writes_raster(tmp_path):
    """auto_correct=True corrects the shifted after raster and records residual shift in provenance."""
    import json
    from geoai04.config import load_config
    from geoai04.pipeline import run_pipeline
    from geoai04.synthetic import generate_synthetic_pair

    syn_dir = tmp_path / "syn_autocorr"
    generate_synthetic_pair(syn_dir, shift_px=(2.0, 0.0))

    out_dir = tmp_path / "out_autocorr"
    res = run_pipeline(
        syn_dir / "before.tif",
        syn_dir / "after.tif",
        load_config(),
        out_root=out_dir,
        run_name="autocorr_test",
        auto_correct=True,
    )

    assert res.status == "completed"
    assert (res.run_dir / "after_aligned.tif").is_file()
    assert res.residual_alignment is not None
    assert res.residual_alignment.magnitude_px < 0.5

    prov = json.loads((res.run_dir / "provenance.json").read_text())
    assert prov["alignment"]["auto_correct_performed"] is True

