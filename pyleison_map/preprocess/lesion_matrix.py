"""
Shared utilities for transforming lesion images into subject-by-voxel matrices with
configurable normalization steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - optional dependency
    sitk = None  # type: ignore

try:
    from ants.core.ants_image import ANTsImage  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    ANTsImage = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ants.core.ants_image import ANTsImage as _ANTsImageType  # type: ignore[attr-defined]
else:
    _ANTsImageType = Any  # type: ignore[misc]

ImageLike = Union[np.ndarray, "sitk.Image", "_ANTsImageType", str, Path]

__all__ = [
    "ImageLike",
    "PreprocessOptions",
    "LesionMatrixResult",
    "build_lesion_matrix",
    "vectorize_image_to_mask",
]


@dataclass
class PreprocessOptions:
    """
    Configuration for image-to-matrix preprocessing.
    """

    mask: Optional[Union[np.ndarray, ImageLike]] = None
    binarize: bool = True
    min_voxel_lesion_count: int = 1
    drop_empty_rows: bool = False
    apply_voxelwise_zscore: bool = False
    apply_subjectwise_l2: bool = False


@dataclass
class LesionMatrixResult:
    """
    Output of `build_lesion_matrix`.
    """

    matrix: np.ndarray
    feature_mask: np.ndarray  # flattened boolean mask
    volume_shape: Tuple[int, int, int]
    kept_indices: np.ndarray
    mask_volume: np.ndarray  # 3D boolean mask
    voxel_means: Optional[np.ndarray] = None
    voxel_stds: Optional[np.ndarray] = None
    row_norms: Optional[np.ndarray] = None


def build_lesion_matrix(
    images: Sequence[ImageLike],
    *,
    mask: Optional[Union[np.ndarray, ImageLike]] = None,
    binarize: bool = True,
    min_voxel_lesion_count: int = 1,
    drop_empty_rows: bool = False,
    apply_voxelwise_zscore: bool = False,
    apply_subjectwise_l2: bool = False,
    dtype: np.dtype = np.float32,
    eps: float = 1e-12,
) -> LesionMatrixResult:
    """
    Convert a sequence of lesion images into a subject-by-voxel matrix.

    Parameters
    ----------
    images:
        Sequence of lesion volumes (numpy arrays, SimpleITK images, ANTs images, or paths).
    mask:
        Optional 3D mask limiting the voxel selection. Can be a numpy array or ANTs/SimpleITK image.
        If omitted, the union of lesions across subjects is used.
    binarize:
        If True, convert each image to 0/1 before aggregation.
    min_voxel_lesion_count:
        Minimum number of subjects lesioned at a voxel for it to be retained.
    drop_empty_rows:
        If True, drop subjects that have no lesioned voxels after masking.
    apply_voxelwise_zscore:
        Whether to z-score each voxel column (mean 0, std 1) across subjects.
    apply_subjectwise_l2:
        Whether to L2-normalize each subject row (dTLVC-style normalization).
    dtype:
        Output dtype for the matrix.
    eps:
        Small constant to avoid division by zero during normalization.
    """

    if not images:
        raise ValueError("`images` must contain at least one entry.")

    arrays = [_load_image_array(im) for im in images]
    volume_shape = arrays[0].shape
    for arr in arrays:
        if arr.shape != volume_shape:
            raise ValueError("All images must share the same shape.")

    if binarize:
        arrays = [(arr > 0).astype(np.float32, copy=False) for arr in arrays]
    else:
        arrays = [arr.astype(np.float32, copy=False) for arr in arrays]

    mask_volume = _prepare_mask(mask, volume_shape, arrays)
    lesion_counts = np.sum([(arr > 0).astype(np.int32, copy=False) for arr in arrays], axis=0)
    valid_mask = mask_volume & (lesion_counts >= int(max(1, min_voxel_lesion_count)))

    feature_mask = valid_mask.reshape(-1)
    if not feature_mask.any():
        raise ValueError("Masking removed all voxels; check parameters.")

    matrix = np.stack([arr.reshape(-1)[feature_mask] for arr in arrays], axis=0).astype(dtype, copy=False)

    row_sums = matrix.sum(axis=1)
    if drop_empty_rows:
        kept_indices = np.where(row_sums > 0)[0]
    else:
        kept_indices = np.arange(matrix.shape[0])
    matrix = matrix[kept_indices]

    voxel_means = voxel_stds = None
    if apply_voxelwise_zscore:
        voxel_means = matrix.mean(axis=0, dtype=np.float64)
        voxel_stds = matrix.std(axis=0, dtype=np.float64)
        voxel_stds = np.where(voxel_stds < eps, 1.0, voxel_stds)
        matrix = ((matrix - voxel_means) / voxel_stds).astype(dtype, copy=False)

    row_norms = None
    if apply_subjectwise_l2:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        matrix = (matrix / norms).astype(dtype, copy=False)
        row_norms = norms.squeeze(1).astype(dtype, copy=False)

    return LesionMatrixResult(
        matrix=matrix,
        feature_mask=feature_mask.astype(bool, copy=False),
        volume_shape=volume_shape,
        kept_indices=kept_indices.astype(np.int64, copy=False),
        mask_volume=valid_mask.astype(bool, copy=False),
        voxel_means=None if voxel_means is None else voxel_means.astype(np.float32, copy=False),
        voxel_stds=None if voxel_stds is None else voxel_stds.astype(np.float32, copy=False),
        row_norms=row_norms,
    )


def vectorize_image_to_mask(
    image: ImageLike,
    *,
    feature_mask: np.ndarray,
    volume_shape: Tuple[int, int, int],
    binarize: bool = True,
    apply_voxelwise_zscore: bool = False,
    voxel_means: Optional[np.ndarray] = None,
    voxel_stds: Optional[np.ndarray] = None,
    apply_subjectwise_l2: bool = False,
    eps: float = 1e-12,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Vectorize a single lesion image using a precomputed feature mask and optional normalization.
    """

    arr = _load_image_array(image)
    if arr.shape != volume_shape:
        raise ValueError(f"Image shape {arr.shape} != expected volume shape {volume_shape}.")
    if binarize:
        arr = (arr > 0).astype(np.float32, copy=False)
    vec = arr.reshape(-1)[feature_mask].astype(np.float32, copy=False)

    if apply_voxelwise_zscore:
        if voxel_means is None or voxel_stds is None:
            raise ValueError("voxel_means and voxel_stds must be provided for z-score transformation.")
        adjusted_stds = np.where(voxel_stds < eps, 1.0, voxel_stds)
        vec = (vec - voxel_means) / adjusted_stds

    if apply_subjectwise_l2:
        norm = np.linalg.norm(vec)
        if norm > eps:
            vec = vec / norm

    return vec.astype(dtype, copy=False)


def _load_image_array(image: ImageLike) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")
        if ANTsImage is not None:
            try:
                import ants  # type: ignore[import]
            except ImportError as exc:  # pragma: no cover - optional dependency missing
                raise RuntimeError(
                    "antspy is required to read images from disk when using ANTs backend."
                ) from exc
            arr = ants.image_read(str(path)).numpy()
        else:
            if sitk is None:
                raise RuntimeError("Neither antspy nor SimpleITK is available to load image paths.")
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # type: ignore[arg-type]
    elif ANTsImage is not None and isinstance(image, ANTsImage):
        arr = image.numpy()
    else:
        if sitk is None or not isinstance(image, sitk.Image):
            raise TypeError(
                "Image must be numpy.ndarray, str/Path to an image, SimpleITK.Image, or ants.core.ANTsImage."
            )
        arr = sitk.GetArrayFromImage(image)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _prepare_mask(
    mask: Optional[Union[np.ndarray, ImageLike]],
    volume_shape: Tuple[int, int, int],
    arrays: Sequence[np.ndarray],
) -> np.ndarray:
    if mask is None:
        union = np.zeros(volume_shape, dtype=bool)
        for arr in arrays:
            union |= (arr > 0)
        return union

    if isinstance(mask, np.ndarray):
        mask_arr = mask
    else:
        mask_arr = _load_image_array(mask)
    if mask_arr.shape != volume_shape:
        raise ValueError("Mask shape mismatch.")
    return mask_arr.astype(bool, copy=False)
