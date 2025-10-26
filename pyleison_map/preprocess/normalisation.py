"""
Tools for normalising (registering) subject images to a common template using antspy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import used only for type hints
    import ants
    from ants.core.ants_image import ANTsImage
else:  # Fallback placeholder to satisfy static checkers without importing ants eagerly
    ANTsImage = object  # type: ignore[misc,assignment]

# Allow callers to refer to the core API from * import
__all__ = ["NormalizationResult", "normalize_to_template"]

ANTsImageLike = Union[str, Path, "ANTsImage"]


@dataclass
class NormalizationResult:
    """
    Container describing the output of a single subject-template registration.

    Attributes
    ----------
    warped_image:
        Subject image after being warped into template space.
    warped_segmentation:
        Optional segmentation warped with nearest-neighbour interpolation.
    fwdtransforms:
        Paths to the forward transforms mapping subject -> template, or None when
        `return_transforms` is False.
    invtransforms:
        Paths to the inverse transforms mapping template -> subject, or None when
        `return_transforms` is False.
    image_output_path:
        Path where the warped image was saved (if requested).
    segmentation_output_path:
        Path where the warped segmentation was saved (if requested).
    source_index:
        Original index within the `images` sequence passed to normalize_to_template.
    source_image_path:
        Path to the original image on disk, if it was provided as a path.
    source_segmentation_path:
        Path to the original segmentation on disk, if provided as a path.
    """

    warped_image: "ANTsImage"
    warped_segmentation: Optional["ANTsImage"]
    fwdtransforms: Optional[Tuple[str, ...]]
    invtransforms: Optional[Tuple[str, ...]]
    image_output_path: Optional[Path]
    segmentation_output_path: Optional[Path]
    source_index: int
    source_image_path: Optional[Path]
    source_segmentation_path: Optional[Path]


def normalize_to_template(
    images: Sequence[ANTsImageLike],
    template: ANTsImageLike,
    segmentations: Optional[Sequence[ANTsImageLike]] = None,
    *,
    save_registered_images: bool = False,
    image_output_dir: Optional[Union[str, Path]] = None,
    save_registered_segmentations: bool = False,
    segmentation_output_dir: Optional[Union[str, Path]] = None,
    registration_type: str = "SyN",
    image_interpolator: str = "lanczosWindowedSinc",
    segmentation_interpolator: str = "nearestNeighbor",
    ants_registration_kwargs: Optional[Dict[str, object]] = None,
    return_transforms: bool = True,
) -> List[NormalizationResult]:
    """
    Register a sequence of subject images (and optional segmentations) to a template.

    Parameters
    ----------
    images:
        Sequence of subject images, provided as file paths or `ants.ANTsImage` objects.
    template:
        Template image to register to, provided as a file path or `ants.ANTsImage`.
    segmentations:
        Optional sequence of segmentations aligned with `images`. Each entry must be a
        file path or `ants.ANTsImage`. When provided, this sequence must match the length
        of `images`.
    save_registered_images:
        Whether to write warped images to disk.
    image_output_dir:
        Directory where warped images should be written when `save_registered_images` is True.
    save_registered_segmentations:
        Whether to write warped segmentations to disk.
    segmentation_output_dir:
        Directory where warped segmentations should be written when
        `save_registered_segmentations` is True.
    registration_type:
        The transform family to use. Passed as `type_of_transform` to `ants.registration`.
    image_interpolator:
        Interpolator to use when resampling intensity images via `ants.apply_transforms`.
        Common options include "linear", "lanczosWindowedSinc", or "nearestNeighbor".
    segmentation_interpolator:
        Interpolator used when resampling segmentation masks. Defaults to nearest
        neighbour to preserve discrete labels.
    ants_registration_kwargs:
        Additional keyword arguments forwarded to `ants.registration`. Values here
        override defaults such as `registration_type`.
    return_transforms:
        Whether to include forward/inverse transform paths in each result. Setting this
        to False avoids keeping potentially large transform metadata in memory.

    Returns
    -------
    List[NormalizationResult]
        One entry per subject. Each result contains the warped images, optional transform
        file paths, and any file destinations used for saving.

    Raises
    ------
    ValueError
        If input lengths mismatch or saving is requested without specifying output dirs.
    ImportError
        If `ants` (antspy) is not installed.
    FileNotFoundError
        When a path-based image/segmentation cannot be located.
    """

    try:
        import ants  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "normalize_to_template requires the antspy library. "
            "Install it with `pip install antspyx`."
        ) from exc

    if not images:
        raise ValueError("`images` must contain at least one entry.")

    segmentations = list(segmentations or [])
    if segmentations and len(segmentations) != len(images):
        raise ValueError(
            "`segmentations` must be the same length as `images` when provided."
        )

    if save_registered_images:
        if image_output_dir is None:
            raise ValueError("`image_output_dir` must be provided when saving images.")
        image_output_dir = Path(image_output_dir)
        image_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        image_output_dir = Path(image_output_dir) if image_output_dir else None

    if save_registered_segmentations:
        if segmentation_output_dir is None:
            raise ValueError(
                "`segmentation_output_dir` must be provided when saving segmentations."
            )
        segmentation_output_dir = Path(segmentation_output_dir)
        segmentation_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        segmentation_output_dir = (
            Path(segmentation_output_dir) if segmentation_output_dir else None
        )

    template_image, _ = _load_ants_image(template, "template")

    reg_kwargs = dict(ants_registration_kwargs or {})
    reg_kwargs.setdefault("type_of_transform", registration_type)

    results: List[NormalizationResult] = []

    for idx, image in enumerate(images):
        moving_image, moving_path = _load_ants_image(image, f"image[{idx}]")
        seg_image = seg_path = None
        if segmentations:
            seg_image, seg_path = _load_ants_image(
                segmentations[idx], f"segmentation[{idx}]"
            )

        reg = ants.registration(
            fixed=template_image,
            moving=moving_image,
            **reg_kwargs,
        )
        fwd_transforms = tuple(reg.get("fwdtransforms", ()))
        inv_transforms = tuple(reg.get("invtransforms", ()))
        stored_fwd_transforms = fwd_transforms if return_transforms else None
        stored_inv_transforms = inv_transforms if return_transforms else None

        # Re-sample explicitly to allow caller-specified interpolators.
        warped_image = ants.apply_transforms(
            fixed=template_image,
            moving=moving_image,
            transformlist=fwd_transforms,
            interpolator=image_interpolator,
        )

        warped_segmentation = None
        if seg_image is not None:
            warped_segmentation = ants.apply_transforms(
                fixed=template_image,
                moving=seg_image,
                transformlist=fwd_transforms,
                interpolator=segmentation_interpolator,
            )

        image_save_path = _maybe_write_image(
            warped_image,
            destination_dir=image_output_dir,
            should_save=save_registered_images,
            source_path=moving_path,
            index=idx,
            prefix="norm_img",
        )
        segmentation_save_path = _maybe_write_image(
            warped_segmentation,
            destination_dir=segmentation_output_dir,
            should_save=save_registered_segmentations,
            source_path=seg_path,
            index=idx,
            prefix="norm_seg",
        )

        results.append(
            NormalizationResult(
                warped_image=warped_image,
                warped_segmentation=warped_segmentation,
                fwdtransforms=stored_fwd_transforms,
                invtransforms=stored_inv_transforms,
                image_output_path=image_save_path,
                segmentation_output_path=segmentation_save_path,
                source_index=idx,
                source_image_path=moving_path,
                source_segmentation_path=seg_path,
            )
        )

    return results


def _load_ants_image(
    image: ANTsImageLike,
    label: str,
) -> Tuple["ANTsImage", Optional[Path]]:
    """
    Load an image provided either as an ANTsImage or a filesystem path.
    """
    try:
        import ants  # type: ignore[import]
        from ants.core.ants_image import ANTsImage  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - handled upstream
        raise ImportError("antspy is required to load images.") from exc

    if isinstance(image, ANTsImage):
        return image, None

    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {label} at {path}")

    return ants.image_read(str(path)), path


def _maybe_write_image(
    image: Optional["ANTsImage"],
    *,
    destination_dir: Optional[Path],
    should_save: bool,
    source_path: Optional[Path],
    index: int,
    prefix: str,
) -> Optional[Path]:
    """
    Write an ANTs image to disk if requested.
    """
    if not should_save or image is None:
        return None

    if destination_dir is None:
        raise ValueError("destination_dir must be provided when saving images.")

    destination_dir.mkdir(parents=True, exist_ok=True)
    output_name = _derive_output_name(source_path, index, prefix)
    output_path = destination_dir / output_name

    import ants  # type: ignore[import]

    ants.image_write(image, str(output_path))
    return output_path


def _derive_output_name(
    source_path: Optional[Path],
    index: int,
    prefix: str,
) -> str:
    """
    Generate an output file name for a saved ANTs image.
    """
    if source_path is not None:
        return source_path.name
    return f"{prefix}_{index:03d}.nii.gz"
