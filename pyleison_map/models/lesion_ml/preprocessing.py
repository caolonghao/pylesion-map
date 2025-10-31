"""
Preprocessing utilities (masking, normalization) shared by lesion ML models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from pyleison_map.preprocess.lesion_matrix import (
    ImageLike,
    PreprocessOptions,
    build_lesion_matrix,
    vectorize_image_to_mask,
)

__all__ = [
    "LesionPreprocessor",
    "preprocessing_kwargs_from_options",
]


@dataclass
class LesionPreprocessor:
    """
    Handles masking, vectorization, and dTLVC normalization for lesion images.
    """

    min_lesion_count: int = 1
    brain_mask: Optional[np.ndarray] = None
    keep_empty_subjects: bool = True
    voxelwise_zscore: bool = False
    subjectwise_l2: bool = True

    vol_shape_: Optional[Tuple[int, int, int]] = None
    feat_mask_: Optional[np.ndarray] = None
    kept_idx_: Optional[np.ndarray] = None
    voxel_means_: Optional[np.ndarray] = None
    voxel_stds_: Optional[np.ndarray] = None

    def fit_transform(
        self,
        imgs: Sequence[ImageLike],
        y: np.ndarray,
        binarize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorize lesion volumes, build feature mask, and optionally apply dTLVC.
        Returns the processed matrix and aligned labels after any row drops.
        """
        result = build_lesion_matrix(
            imgs,
            mask=self.brain_mask,
            binarize=binarize,
            min_voxel_lesion_count=self.min_lesion_count,
            drop_empty_rows=not self.keep_empty_subjects,
            apply_voxelwise_zscore=self.voxelwise_zscore,
            apply_subjectwise_l2=self.subjectwise_l2,
            dtype=np.float32,
        )

        self.vol_shape_ = result.volume_shape
        self.feat_mask_ = result.feature_mask
        self.kept_idx_ = result.kept_indices
        self.voxel_means_ = result.voxel_means
        self.voxel_stds_ = result.voxel_stds

        return result.matrix.astype(np.float32, copy=False), y[result.kept_indices]

    def transform_single(self, img: ImageLike, binarize: bool = True) -> np.ndarray:
        """Vectorize one image (numpy/SimpleITK/ANTs/path) and apply stored preprocessing."""
        if self.vol_shape_ is None or self.feat_mask_ is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first.")
        return vectorize_image_to_mask(
            img,
            feature_mask=self.feat_mask_,
            volume_shape=self.vol_shape_,
            binarize=binarize,
            apply_voxelwise_zscore=self.voxelwise_zscore,
            voxel_means=self.voxel_means_,
            voxel_stds=self.voxel_stds_,
            apply_subjectwise_l2=self.subjectwise_l2,
        )

    def vector_to_volume(self, vec: np.ndarray) -> np.ndarray:
        """Scatter a feature vector back to 3D using the stored feature mask."""
        if self.vol_shape_ is None or self.feat_mask_ is None:
            raise RuntimeError("Preprocessor is not fitted.")
        vol = np.zeros(int(np.prod(self.vol_shape_)), dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("vec must be 1D")
        vol[self.feat_mask_] = vec.astype(np.float32, copy=False)
        return vol.reshape(self.vol_shape_)


def preprocessing_kwargs_from_options(
    options: PreprocessOptions,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Map high-level `PreprocessOptions` into keyword arguments accepted by the model factory.
    """
    kwargs: Dict[str, Any] = {} if overrides is None else dict(overrides)
    kwargs.setdefault("min_lesion_count", options.min_voxel_lesion_count)
    kwargs.setdefault("brain_mask", options.mask)
    kwargs.setdefault("keep_empty_subjects", not options.drop_empty_rows)
    kwargs.setdefault("voxelwise_zscore", options.apply_voxelwise_zscore)
    kwargs.setdefault("subjectwise_l2", options.apply_subjectwise_l2)
    return kwargs
