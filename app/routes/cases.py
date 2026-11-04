"""Case endpoints."""

from __future__ import annotations

import shutil
import uuid
from io import BytesIO
from pathlib import Path
from typing import Annotated

import nibabel as nib
import numpy as np
from PIL import Image

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response

from app.config import settings
from app.schemas import CaseCreate, CaseRead
from app.services import CaseService


router = APIRouter(prefix="/cases", tags=["Cases"])
case_service = CaseService()


def _save_modality(
    upload_dir: Path,
    case_name: str,
    suffix: str,
    file: UploadFile,
) -> None:
    """Save an uploaded MRI (or ground-truth segmentation) file."""
    filename = file.filename or ""
    ext = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"

    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in case_name
    ).strip("_") or "case"

    destination = upload_dir / f"{safe_name}_{suffix}{ext}"

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


def _case_directory(case) -> Path:
    """Find the directory containing the case's MRI files."""
    raw_path = getattr(case, "nifti_path", None)

    if not raw_path:
        raise HTTPException(
            status_code=500,
            detail="The case record does not contain an MRI directory path.",
        )

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.is_dir():
        raise HTTPException(
            status_code=404,
            detail="The uploaded MRI directory could not be found.",
        )

    return path


def _find_mri_file(directory: Path, modality: str) -> Path:
    """Find an uploaded MRI file by modality."""
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (
            path.name.lower().endswith(f"_{modality}.nii")
            or path.name.lower().endswith(f"_{modality}.nii.gz")
        )
    )

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded {modality.upper()} NIfTI file was found.",
        )

    return matches[0]


def _find_groundtruth_file(directory: Path) -> Path | None:
    """Find the optional uploaded ground-truth segmentation, if any."""
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (
            path.name.lower().endswith("_groundtruth.nii")
            or path.name.lower().endswith("_groundtruth.nii.gz")
        )
    )
    return matches[0] if matches else None


def _load_nifti(path: Path):
    """Load a NIfTI image and its voxel data."""
    try:
        image = nib.load(str(path))
        volume = np.asarray(image.get_fdata(), dtype=np.float32)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read NIfTI file {path.name}: {exc}",
        ) from exc

    if volume.ndim != 3:
        raise HTTPException(
            status_code=422,
            detail=f"{path.name} must contain a 3D NIfTI volume.",
        )

    if not np.isfinite(volume).any():
        raise HTTPException(
            status_code=422,
            detail=f"{path.name} contains no valid voxel values.",
        )

    return image, volume


def _normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """Convert an MRI slice into an 8-bit grayscale image."""
    slice_data = np.asarray(slice_data, dtype=np.float32)
    valid = slice_data[np.isfinite(slice_data)]

    if valid.size == 0:
        return np.zeros(slice_data.shape, dtype=np.uint8)

    nonzero = valid[valid != 0]

    if nonzero.size > 0:
        low, high = np.percentile(nonzero, [1, 99])
    else:
        low, high = np.percentile(valid, [1, 99])

    if high <= low:
        return np.zeros(slice_data.shape, dtype=np.uint8)

    normalized = (slice_data - low) / (high - low) * 255.0

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    return np.clip(normalized, 0, 255).astype(np.uint8)


def _orient_for_display(slice_data: np.ndarray) -> np.ndarray:
    """Rotate a slice for display."""
    return np.rot90(slice_data)


def _png_response(rgb: np.ndarray) -> Response:
    """Return an RGB image as a PNG response."""
    output = BytesIO()

    Image.fromarray(rgb.astype(np.uint8)).save(
        output,
        format="PNG",
    )

    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


def _find_prediction_mask(directory: Path, case_id: int) -> Path:
    """Find the model's saved segmentation mask for a case.

    The inference service saves predictions to
    ``settings.results_dir / case_<id> / mask.nii.gz`` (not in the upload
    directory), so look there first. Fall back to searching the upload
    directory for older layouts. The optional ground-truth upload is saved as
    ``*_groundtruth.nii(.gz)`` and is never treated as a prediction.
    """
    saved = settings.results_dir / f"case_{case_id}" / "mask.nii.gz"
    if saved.is_file():
        return saved

    keywords = (
        "mask",
        "segmentation",
        "seg",
        "prediction",
        "pred",
    )

    candidates = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name.lower().endswith((".nii", ".nii.gz"))
        and "groundtruth" not in path.name.lower()
        and any(word in path.name.lower() for word in keywords)
        and not any(
            path.name.lower().endswith(f"_{modality}{extension}")
            for modality in ("t1ce", "t2", "flair")
            for extension in (".nii", ".nii.gz")
        )
    )

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                "No saved segmentation mask was found for this case. "
                "Run the prediction first."
            ),
        )

    return candidates[0]


def _select_slice(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
) -> int:
    """Select a useful slice, preferring slices containing mask voxels."""
    depth = volume.shape[2]
    middle = depth // 2

    if mask is not None:
        if mask.shape != volume.shape:
            return middle

        counts = np.count_nonzero(
            np.isfinite(mask) & (mask > 0),
            axis=(0, 1),
        )

        if counts.max() > 0:
            best_slices = np.flatnonzero(counts == counts.max())
            return int(best_slices[np.argmin(np.abs(best_slices - middle))])

    scores = []

    for index in range(depth):
        current_slice = volume[:, :, index]
        valid = current_slice[np.isfinite(current_slice)]

        if valid.size == 0:
            scores.append(0.0)
            continue

        occupancy = np.count_nonzero(valid) / valid.size

        if valid.size > 1:
            contrast = float(np.percentile(valid, 95) - np.percentile(valid, 5))
        else:
            contrast = 0.0

        scores.append(occupancy * contrast)

    best_score = max(scores, default=0.0)

    if best_score <= 0:
        return middle

    best_slices = [
        index
        for index, score in enumerate(scores)
        if score == best_score
    ]

    return int(min(best_slices, key=lambda index: abs(index - middle)))


@router.post("/", response_model=CaseRead, status_code=201)
async def create_case(
    case_name: Annotated[str, Form()],
    patient_id: Annotated[str | None, Form()] = None,
    modality_summary: Annotated[str | None, Form()] = None,
    t1ce_file: Annotated[UploadFile | None, File()] = None,
    t2_file: Annotated[UploadFile | None, File()] = None,
    flair_file: Annotated[UploadFile | None, File()] = None,
    seg_file: Annotated[UploadFile | None, File()] = None,
):
    """Upload the three required MRI modalities.

    ``seg_file`` is an OPTIONAL ground-truth segmentation (for example a
    BraTS ``*_seg.nii.gz``) used only to compare against the model's output.
    """
    if not (t1ce_file and t2_file and flair_file):
        raise HTTPException(
            status_code=400,
            detail=(
                "All three files are required: "
                "t1ce_file, t2_file, and flair_file."
            ),
        )

    case_id_tmp = str(uuid.uuid4())
    upload_dir = settings.upload_dir / case_id_tmp
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = case_name.replace(" ", "_").strip("_") or "case"

    try:
        _save_modality(upload_dir, safe_name, "t1ce", t1ce_file)
        _save_modality(upload_dir, safe_name, "t2", t2_file)
        _save_modality(upload_dir, safe_name, "flair", flair_file)

        # Browsers send an empty part when no file is chosen; skip those.
        if seg_file is not None and (seg_file.filename or "").strip():
            _save_modality(upload_dir, safe_name, "groundtruth", seg_file)

        case_data = CaseCreate(
            case_name=case_name,
            patient_id=patient_id,
            modality_summary=modality_summary,
        )

        return case_service.create_case(case_data, str(upload_dir))

    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    finally:
        await t1ce_file.close()
        await t2_file.close()
        await flair_file.close()
        if seg_file is not None:
            await seg_file.close()


@router.get("/", response_model=list[CaseRead])
async def list_cases():
    """List all MRI cases."""
    return case_service.list_all_cases()


@router.get("/{case_id}/preview/{modality}")
async def preview_mri(case_id: int, modality: str):
    """Return an axial PNG preview of an MRI or saved segmentation mask."""
    modality = modality.lower()

    if modality not in {"t1ce", "t2", "flair", "segmentation"}:
        raise HTTPException(
            status_code=400,
            detail="Modality must be t1ce, t2, flair, or segmentation.",
        )

    case = case_service.get_case_by_id(case_id)

    if not case:
        raise HTTPException(
            status_code=404,
            detail="Case not found.",
        )

    directory = _case_directory(case)

    if modality == "segmentation":
        mask_path = _find_prediction_mask(directory, case_id)
        mask_image, mask_volume = _load_nifti(mask_path)

        t1ce_path = _find_mri_file(directory, "t1ce")
        mri_image, mri_volume = _load_nifti(t1ce_path)

        if mask_volume.shape != mri_volume.shape:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The segmentation mask and T1ce MRI have different shapes. "
                    "They must be aligned before an overlay can be displayed."
                ),
            )

        if not np.allclose(
            mask_image.affine,
            mri_image.affine,
            atol=1e-3,
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "The segmentation mask and T1ce MRI have different "
                    "spatial orientations or positions. Align them before "
                    "displaying an overlay."
                ),
            )

        slice_index = _select_slice(mri_volume, mask_volume)

        background = _normalize_slice(
            _orient_for_display(mri_volume[:, :, slice_index])
        )

        mask_slice = _orient_for_display(mask_volume[:, :, slice_index])

        rgb = np.stack(
            [background, background, background],
            axis=-1,
        )

        tumor = np.isfinite(mask_slice) & (mask_slice > 0)

        rgb[tumor, 0] = 255
        rgb[tumor, 1] = (
            rgb[tumor, 1].astype(np.float32) * 0.20
        ).astype(np.uint8)
        rgb[tumor, 2] = (
            rgb[tumor, 2].astype(np.float32) * 0.20
        ).astype(np.uint8)

        return _png_response(rgb)

    image_path = _find_mri_file(directory, modality)
    _, volume = _load_nifti(image_path)

    slice_index = _select_slice(volume)
    image_slice = _orient_for_display(volume[:, :, slice_index])
    grayscale = _normalize_slice(image_slice)

    rgb = np.stack(
        [grayscale, grayscale, grayscale],
        axis=-1,
    )

    return _png_response(rgb)


@router.get("/{case_id}/dice")
async def dice_score(case_id: int):
    """Whole-tumor Dice between the model's mask and the uploaded ground truth."""
    case = case_service.get_case_by_id(case_id)

    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    directory = _case_directory(case)

    gt_path = _find_groundtruth_file(directory)
    if gt_path is None:
        raise HTTPException(
            status_code=404,
            detail="No ground-truth segmentation was uploaded for this case.",
        )

    pred_path = _find_prediction_mask(directory, case_id)

    _, gt_volume = _load_nifti(gt_path)
    _, pred_volume = _load_nifti(pred_path)

    if gt_volume.shape != pred_volume.shape:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Ground truth {gt_volume.shape} and prediction "
                f"{pred_volume.shape} have different shapes, so Dice cannot "
                "be computed."
            ),
        )

    gt = gt_volume > 0
    pred = pred_volume > 0

    gt_voxels = int(gt.sum())
    pred_voxels = int(pred.sum())
    intersection = int(np.logical_and(gt, pred).sum())

    denominator = gt_voxels + pred_voxels
    dice = (2.0 * intersection / denominator) if denominator else None

    return {
        "case_id": case_id,
        "dice_whole_tumor": None if dice is None else round(dice, 4),
        "ground_truth_voxels": gt_voxels,
        "predicted_voxels": pred_voxels,
        "overlap_voxels": intersection,
    }


@router.get("/{case_id}", response_model=CaseRead)
async def get_case(case_id: int):
    """Get details for a specific case."""
    case = case_service.get_case_by_id(case_id)

    if not case:
        raise HTTPException(
            status_code=404,
            detail="Case not found.",
        )

    return case