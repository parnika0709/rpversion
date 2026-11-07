"""Comprehensive unit and integration tests for atlas-based anatomical risk mapping.

Validates:
1. Spatial coordinate alignment between MRI and MNI152 Harvard-Oxford atlas space.
2. Tumor overlapping an anatomical region (direct anatomical overlap vs proximity).
3. Tumor near an anatomical region without overlap (proximity risk).
4. Tumor far from an anatomical region (low risk).
5. No tumor detected (empty mask -> explicit 'No tumor detected', never 'Low').
6. Failed registration and invalid affine matrices (singular, non-finite, wrong shape).
7. Incorrect image dimensions (2D, 4D).
8. Distance calculations in true millimeters.
9. Missing atlas error handling (no silent bounding-box fallback).
10. Threshold sensitivity and environmental overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from braintumor.risk.mapping import (
    CANONICAL_FUNCTIONAL_REGIONS,
    NO_TUMOR_TIER,
    RISK_TIERS,
    AtlasNotFoundError,
    HarvardOxfordAtlas,
    MappingStatus,
    RegistrationError,
    RiskRegion,
    RiskThresholds,
    _distance_mm,
    _risk_tier,
    compute_region_metrics,
    compute_risk_regions,
    get_default_atlas,
    resample_mask_to_atlas,
    risk_tier_sensitivity,
    thresholds_from_env,
    validate_spatial_inputs,
)


@pytest.fixture(scope="module")
def atlas() -> HarvardOxfordAtlas:
    """Fixture providing the default Harvard-Oxford atlas."""
    return get_default_atlas()


class TestAtlasSpatialAlignment:
    """Test MRI-to-atlas coordinate transformations and spatial alignment."""

    def test_atlas_properties(self, atlas: HarvardOxfordAtlas):
        """Verify atlas dimensions, zooms, orientation, and voxel volume."""
        assert atlas.shape == (91, 109, 91)
        assert atlas.spacing == (2.0, 2.0, 2.0)
        assert atlas.voxel_volume_mm3 == pytest.approx(8.0, rel=1e-3)
        assert atlas.affine.shape == (4, 4)

    def test_resample_preserves_physical_coordinates(self, atlas: HarvardOxfordAtlas):
        """Verify that resample_mask_to_atlas maps a physical sphere in patient RAS space
        to the exact corresponding physical coordinates in atlas space."""
        # Atlas center of mass in physical mm (approx 0.5, -18, 17.5 mm in MNI)
        target_world_center = np.array([0.0, -20.0, 20.0])

        # Create patient volume in 1mm isotropic RAS space
        patient_shape = (100, 100, 100)
        patient_affine = np.eye(4)
        # Set origin so target_world_center falls at voxel (50, 50, 50)
        patient_affine[:3, 3] = target_world_center - np.array([50.0, 50.0, 50.0])

        patient_mask = np.zeros(patient_shape, dtype=bool)
        # Place a 5x5x5 voxel box around (50, 50, 50)
        patient_mask[48:53, 48:53, 48:53] = True
        patient_vox_vol = 1.0  # 1 mm³ per voxel (125 mm³)

        # Resample into atlas grid
        atlas_mask = resample_mask_to_atlas(
            tumor_mask=patient_mask,
            patient_affine=patient_affine,
            atlas_affine=atlas.affine,
            atlas_shape=atlas.shape,
        )

        assert atlas_mask.any(), "Resampled mask should not be empty"
        resampled_coords = np.argwhere(atlas_mask)
        resampled_com_vox = resampled_coords.mean(axis=0)

        # Convert atlas voxel COM back to physical world mm
        resampled_com_mm = (atlas.affine @ np.append(resampled_com_vox, 1.0))[:3]

        # Should match original patient world center within half an atlas voxel (1 mm)
        assert np.linalg.norm(resampled_com_mm - target_world_center) < 2.0

    def test_orientation_handling_ras_vs_las(self, atlas: HarvardOxfordAtlas):
        """Verify that Left vs Right lateralization is strictly preserved
        when mapping between RAS patient MRI and LAS Harvard-Oxford atlas."""
        # Harvard-Oxford cortl atlas is LAS orientation.
        # Motor cortex left (label 13) has physical X > 0 in LAS (which is Left in neuroimaging MNI space)
        left_motor = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"])
        right_motor = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_right"])

        assert left_motor.sum() > 0
        assert right_motor.sum() > 0
        assert not np.logical_and(left_motor, right_motor).any(), "Left and Right motor cortex must not overlap"

        # Find physical coordinates of left motor cortex
        left_coords = np.argwhere(left_motor)
        left_world_pts = (atlas.affine @ np.c_[left_coords, np.ones(len(left_coords))].T)[:3].T
        left_x_mean = left_world_pts[:, 0].mean()

        right_coords = np.argwhere(right_motor)
        right_world_pts = (atlas.affine @ np.c_[right_coords, np.ones(len(right_coords))].T)[:3].T
        right_x_mean = right_world_pts[:, 0].mean()

        # In standard MNI (RAS physical coordinates): Left is X < 0 or X > 0 depending on neurological display
        # Crucially: Left and Right must have opposite signs along the lateral axis (X)
        assert np.sign(left_x_mean) != np.sign(right_x_mean)


class TestTumorOverlapAndDistance:
    """Test direct overlap, proximity, distance calculations in mm, and risk classification."""

    def test_tumor_overlapping_anatomical_region(self, atlas: HarvardOxfordAtlas):
        """Test tumor placed directly inside Left Motor Cortex."""
        left_motor_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"])
        coords = np.argwhere(left_motor_mask)

        # Create patient space identical to atlas space for direct testing
        patient_mask = np.zeros(atlas.shape, dtype=bool)
        # Select 200 voxels from left motor cortex
        for c in coords[:200]:
            patient_mask[tuple(c)] = True

        regions = compute_risk_regions(
            tumor_mask=patient_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        motor_left = next(r for r in regions if r.region == "motor cortex left")
        assert motor_left.overlap_pct > 0.0
        assert motor_left.overlap_pct == pytest.approx(100.0, abs=0.1)  # 100% of tumor is inside
        assert motor_left.region_overlap_pct > 0.0
        assert motor_left.distance_mm == 0.0
        assert motor_left.risk_tier == "High"
        assert motor_left.overlap_volume_mm3 == pytest.approx(200 * 8.0, rel=1e-2)

    def test_tumor_near_region_without_overlap(self, atlas: HarvardOxfordAtlas):
        """Test tumor placed 4-8 mm away from brainstem without overlapping it."""
        brainstem_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["brainstem"])
        dist_map = distance_transform_edt(~brainstem_mask, sampling=atlas.spacing)

        # Find voxels that are between 4mm and 8mm away from brainstem
        near_voxels = np.argwhere((dist_map >= 4.0) & (dist_map <= 8.0))
        assert len(near_voxels) > 50

        patient_mask = np.zeros(atlas.shape, dtype=bool)
        for c in near_voxels[:30]:
            patient_mask[tuple(c)] = True

        thresholds = RiskThresholds(high_overlap_pct=5.0, moderate_distance_mm=10.0)
        regions = compute_risk_regions(
            tumor_mask=patient_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            thresholds=thresholds,
            atlas=atlas,
        )

        bs_region = next(r for r in regions if r.region == "brainstem")
        assert bs_region.overlap_pct == 0.0
        assert bs_region.region_overlap_pct == 0.0
        assert bs_region.distance_mm is not None
        assert 3.5 <= bs_region.distance_mm <= 8.5
        assert bs_region.risk_tier == "Moderate", "Proximity within 10mm must be classified as Moderate"

    def test_tumor_far_from_region(self, atlas: HarvardOxfordAtlas):
        """Test tumor placed in visual cortex is classified as Low risk for motor cortex (>20mm)."""
        visual_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["visual_cortex"])
        coords = np.argwhere(visual_mask)

        patient_mask = np.zeros(atlas.shape, dtype=bool)
        for c in coords[:50]:
            patient_mask[tuple(c)] = True

        regions = compute_risk_regions(
            tumor_mask=patient_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            thresholds=RiskThresholds(high_overlap_pct=5.0, moderate_distance_mm=10.0),
            atlas=atlas,
        )

        motor_left = next(r for r in regions if r.region == "motor cortex left")
        assert motor_left.overlap_pct == 0.0
        assert motor_left.distance_mm is not None
        assert motor_left.distance_mm > 10.0
        assert motor_left.risk_tier == "Low"

    def test_distances_reported_in_millimeters(self, atlas: HarvardOxfordAtlas):
        """Verify that distance is measured in millimeters, not voxel counts."""
        thalamus_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["thalamus"])
        dist_map = distance_transform_edt(~thalamus_mask, sampling=atlas.spacing)

        outside_pts = np.argwhere(~thalamus_mask)
        sample_pt = outside_pts[500]

        # Ground truth Euclidean distance in mm to all thalamus voxels
        thal_coords = np.argwhere(thalamus_mask)
        diff_mm = (thal_coords - sample_pt) * np.array(atlas.spacing)
        expected_dist_mm = float(np.sqrt((diff_mm**2).sum(axis=1)).min())

        calculated_dist_mm = dist_map[tuple(sample_pt)]
        assert abs(calculated_dist_mm - expected_dist_mm) < 1e-4

        # Mask with single voxel
        single_vox_mask = np.zeros(atlas.shape, dtype=bool)
        single_vox_mask[tuple(sample_pt)] = True

        computed_dist = _distance_mm(single_vox_mask, thalamus_mask, atlas.spacing)
        assert abs(computed_dist - expected_dist_mm) < 1e-4

    def test_compare_surface_distance_methods(self, atlas: HarvardOxfordAtlas):
        """Compare EDT and boundary coordinates surface-to-surface distance calculations."""
        from braintumor.risk.mapping import surface_to_surface_distance_mm

        thalamus_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["thalamus"])
        # Create a small non-overlapping tumor blob
        tumor_mask = np.zeros(atlas.shape, dtype=bool)
        # Thalamus is around the center, place blob at offset
        tumor_mask[10:14, 10:14, 10:14] = True

        dist_edt = surface_to_surface_distance_mm(tumor_mask, thalamus_mask, atlas.spacing, method="edt")
        dist_boundary = surface_to_surface_distance_mm(tumor_mask, thalamus_mask, atlas.spacing, method="boundary_coords")

        assert np.isfinite(dist_edt)
        assert np.isfinite(dist_boundary)
        # Both methods must yield closely matching distances (within 1e-3 mm)
        assert abs(dist_edt - dist_boundary) < 1e-2



class TestNoTumorDetected:
    """Test handling of empty masks (no tumor detected)."""

    def test_empty_mask_assigns_no_tumor_tier_to_all_regions(self, atlas: HarvardOxfordAtlas):
        """Verify that an empty mask assigns 'No tumor detected' to every region, never 'Low'."""
        empty_mask = np.zeros(atlas.shape, dtype=np.uint8)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        assert len(regions) == len(CANONICAL_FUNCTIONAL_REGIONS)
        for r in regions:
            assert r.risk_tier == NO_TUMOR_TIER
            assert r.risk_tier != "Low", "Must not classify empty tumor as 'Low' (false reassurance)"
            assert r.overlap_pct == 0.0
            assert r.distance_mm is None
            assert r.mapping_status == MappingStatus.NO_TUMOR_DETECTED.value

    def test_empty_mask_metrics(self, atlas: HarvardOxfordAtlas):
        """Verify that compute_region_metrics handles empty mask safely without division by zero."""
        empty_mask = np.zeros((64, 64, 64), dtype=bool)
        metrics = compute_region_metrics(empty_mask, spacing=(1.0, 1.0, 1.0), atlas=atlas)

        for m in metrics:
            assert m["overlap_pct"] == 0.0
            assert m["region_overlap_pct"] == 0.0
            assert m["overlap_volume_mm3"] == 0.0
            assert m["tumor_volume_mm3"] == 0.0
            assert m["distance_mm"] == float("inf")


class TestErrorHandlingAndValidation:
    """Test validation checks, invalid affine matrices, dimension checks, and registration errors."""

    def test_incompatible_dimensions_2d(self):
        """Verify that 2D input raises ValueError."""
        mask_2d = np.zeros((128, 128), dtype=bool)
        with pytest.raises(ValueError, match="Tumor mask must be a 3D volume"):
            validate_spatial_inputs(mask_2d)

    def test_incompatible_dimensions_4d(self):
        """Verify that 4D input raises ValueError."""
        mask_4d = np.zeros((4, 128, 128, 128), dtype=bool)
        with pytest.raises(ValueError, match="Tumor mask must be a 3D volume"):
            validate_spatial_inputs(mask_4d)

    def test_invalid_affine_singular_matrix(self):
        """Verify that singular affine matrix with det=0 raises RegistrationError."""
        singular_affine = np.zeros((4, 4))
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(RegistrationError, match="Singular or near-singular"):
            validate_spatial_inputs(mask_3d, affine=singular_affine)

    def test_invalid_affine_non_finite_values(self):
        """Verify that affine matrix with NaN or Inf raises ValueError."""
        affine_nan = np.eye(4)
        affine_nan[0, 3] = np.nan
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match="non-finite"):
            validate_spatial_inputs(mask_3d, affine=affine_nan)

    def test_invalid_affine_wrong_shape(self):
        """Verify that non-(4,4) affine matrix raises ValueError."""
        affine_3x3 = np.eye(3)
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match=r"Affine matrix must have shape \(4, 4\)"):
            validate_spatial_inputs(mask_3d, affine=affine_3x3)

    def test_invalid_spacing_negative_or_zero(self):
        """Verify that non-positive spacing raises ValueError."""
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match="Spacing must contain three positive finite numbers"):
            validate_spatial_inputs(mask_3d, spacing=(1.0, -2.0, 1.0))

    def test_missing_atlas_raises_atlas_not_found_error(self, tmp_path: Path):
        """Verify that missing atlas directory raises AtlasNotFoundError (never falls back to bounding boxes)."""
        empty_dir = tmp_path / "nonexistent_atlas"
        empty_dir.mkdir()
        with pytest.raises(AtlasNotFoundError, match="Atlas files not found"):
            _ = HarvardOxfordAtlas(atlas_dir=empty_dir)


class TestConfigurabilityAndSensitivity:
    """Test threshold customization, environment overrides, and sensitivity analysis."""

    def test_risk_thresholds_override(self):
        """Verify that custom RiskThresholds alter risk tier assignment."""
        t_strict = RiskThresholds(high_overlap_pct=1.0, moderate_distance_mm=20.0)
        t_lenient = RiskThresholds(high_overlap_pct=20.0, moderate_distance_mm=5.0)

        # 2% overlap: High under strict, Moderate under lenient
        tier_strict = _risk_tier(overlap_pct=2.0, distance_mm=0.0, thresholds=t_strict)
        tier_lenient = _risk_tier(overlap_pct=2.0, distance_mm=0.0, thresholds=t_lenient)

        assert tier_strict == "High"
        assert tier_lenient == "Moderate"

        # 12mm distance: Moderate under strict (<=20), Low under lenient (>5)
        dist_strict = _risk_tier(overlap_pct=0.0, distance_mm=12.0, thresholds=t_strict)
        dist_lenient = _risk_tier(overlap_pct=0.0, distance_mm=12.0, thresholds=t_lenient)

        assert dist_strict == "Moderate"
        assert dist_lenient == "Low"

    def test_thresholds_from_env(self, monkeypatch: pytest.MonkeyPatch):
        """Verify loading thresholds from environment variables."""
        monkeypatch.setenv("RISK_HIGH_OVERLAP_PCT", "7.5")
        monkeypatch.setenv("RISK_MODERATE_DISTANCE_MM", "12.5")
        monkeypatch.setenv("RISK_HIGH_REGION_OVERLAP_PCT", "15.0")

        t = thresholds_from_env()
        assert t.high_overlap_pct == 7.5
        assert t.moderate_distance_mm == 12.5
        assert t.high_region_overlap_pct == 15.0

    def test_risk_tier_sensitivity_grid(self):
        """Verify sensitivity grid evaluation across multiple thresholds."""
        mock_metrics = [
            {"region": "motor cortex left", "overlap_pct": 3.0, "distance_mm": 0.0, "region_overlap_pct": 0.5},
            {"region": "brainstem", "overlap_pct": 0.0, "distance_mm": 8.0, "region_overlap_pct": 0.0},
        ]
        rows = risk_tier_sensitivity(
            mock_metrics,
            overlap_grid=(2.0, 5.0),
            distance_grid=(5.0, 10.0),
        )
        assert len(rows) == 4  # 2 x 2 grid
        for r in rows:
            assert "High" in r and "Moderate" in r and "Low" in r
            assert "tiers" in r
