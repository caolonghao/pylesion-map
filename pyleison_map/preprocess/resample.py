"""
Resampling utilities for bringing images to common spacing or shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import used only for type hints
    from ants.core.ants_image import ANTsImage
else:
    ANTsImage = object  # type: ignore[misc,assignment]

ANTsImageLike = Union[str, Path, "ANTsImage"]

__all__ = ["ResampleResult", "resample_images"]


_INTERP_TO_CODE: Dict[str, int] = {
    "linear": 0,
    "nearestneighbor": 1,
    "nearest_neighbor": 1,
    "nearest": 1,
    "bspline": 3,
}


@dataclass
class ResampleResult:
    """
    Describes the outcome of resampling a single input image.

    Attributes
    ----------
    resampled_image:
        The image after resampling.
    spacing:
        Tuple describing voxel spacing of the resampled image.
    shape:
        Tuple describing voxel counts of the resampled image.
    output_path:
        Where the resampled image was saved (if requested).
    source_index:
        Index of the image within the provided `images` sequence.
    source_path:
        Path to the original image if it was supplied as such.
    """

    resampled_image: "ANTsImage"
    spacing: Tuple[float, ...]
    shape: Tuple[int, ...]
    output_path: Optional[Path]
    source_index: int
    source_path: Optional[Path]


def resample_images(
    images: Sequence[ANTsImageLike],
    *,
    target_spacing: Optional[Sequence[float]] = None,
    target_shape: Optional[Sequence[int]] = None,
    save_resampled: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    interpolator: str = "linear",
    ants_resample_kwargs: Optional[Dict[str, object]] = None,
) -> List[ResampleResult]:
    """
    Resample images to a target spacing and/or voxel shape using antspy.

    Parameters
    ----------
    images:
        Sequence of images as `ants.ANTsImage` objects or filesystem paths.
    target_spacing:
        Optional sequence representing the desired voxel spacing
        (e.g. [1.0, 1.0, 1.0]). Length must match image dimensionality or be 1
        (broadcasted to all axes).
    target_shape:
        Optional sequence representing the desired output voxel counts
        (e.g. [256, 256, 128]). Length must match image dimensionality or be 1.
    save_resampled:
        If True, write each resampled image to `output_dir`.
    output_dir:
        Destination directory used when `save_resampled` is True.
    interpolator:
        Interpolation strategy ("linear", "nearest", "bspline").
    ants_resample_kwargs:
        Additional arguments forwarded to `ants.resample_image`.

    Returns
    -------
    List[ResampleResult]
        Per-image results, including the resampled image (kept in memory if not saved).

    Raises
    ------
    ValueError
        If neither `target_spacing` nor `target_shape` is provided.
    ImportError
        If antspy is unavailable.
    FileNotFoundError
        If a path-based image does not exist.
    """

    if target_spacing is None and target_shape is None:
        raise ValueError("Specify at least one of `target_spacing` or `target_shape`.")

    try:
        import ants  # type: ignore[import]
        from ants.core.ants_image import ANTsImage  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "resample_images requires the antspy library. Install it via `pip install antspyx`."
        ) from exc

    interp_code = _resolve_interpolator(interpolator)

    if save_resampled:
        if output_dir is None:
            raise ValueError("`output_dir` must be provided when saving resampled data.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir) if output_dir else None

    resample_kwargs = dict(ants_resample_kwargs or {})
    for forbidden_key in ("resample_params", "use_voxels", "interp_type"):
        resample_kwargs.pop(forbidden_key, None)

    results: List[ResampleResult] = []

    for idx, image in enumerate(images):
        ants_image, source_path = _load_ants_image(image, f"image[{idx}]")
        dimension = ants_image.dimension

        resampled = ants_image

        if target_spacing is not None:
            spacing_params = _prepare_numeric_sequence(
                target_spacing,
                dimension,
                label="target_spacing",
                coerce_type=float,
            )
            resampled = ants.resample_image(
                image=resampled,
                resample_params=spacing_params,
                use_voxels=False,
                interp_type=interp_code,
                **resample_kwargs,
            )

        if target_shape is not None:
            shape_params = _prepare_numeric_sequence(
                target_shape,
                dimension,
                label="target_shape",
                coerce_type=int,
            )
            resampled = ants.resample_image(
                image=resampled,
                resample_params=shape_params,
                use_voxels=True,
                interp_type=interp_code,
                **resample_kwargs,
            )

        output_path = _maybe_write_image(
            resampled,
            destination_dir=output_dir,
            should_save=save_resampled,
            source_path=source_path,
            index=idx,
            prefix="resamp",
        )

        results.append(
            ResampleResult(
                resampled_image=resampled,
                spacing=tuple(resampled.spacing),
                shape=tuple(resampled.shape),
                output_path=output_path,
                source_index=idx,
                source_path=source_path,
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
    image: "ANTsImage",
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
    if not should_save:
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
    Generate an output filename for saved resampled images.
    """
    if source_path is not None:
        return source_path.name
    return f"{prefix}_{index:03d}.nii.gz"


def _prepare_numeric_sequence(
    values: Sequence[Union[int, float]],
    dimension: int,
    *,
    label: str,
    coerce_type,
) -> Tuple[Union[int, float], ...]:
    """
    Validate and broadcast numeric sequences to match image dimensionality.
    """
    if not values:
        raise ValueError(f"{label} must be a non-empty sequence.")

    if len(values) == 1:
        values = [values[0]] * dimension
    elif len(values) != dimension:
        raise ValueError(
            f"{label} must have length 1 or match image dimension ({dimension})."
        )

    try:
        converted = tuple(coerce_type(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} entries must be numeric.") from exc

    return converted


def _resolve_interpolator(name: str) -> int:
    """
    Convert textual interpolation names to the integer codes expected by antspy.
    """
    key = name.replace(" ", "").replace("-", "").lower()
    if key not in _INTERP_TO_CODE:
        raise ValueError(
            f"Unsupported interpolator '{name}'. "
            f"Supported: {', '.join(sorted(_INTERP_TO_CODE))}."
        )
    return _INTERP_TO_CODE[key]
