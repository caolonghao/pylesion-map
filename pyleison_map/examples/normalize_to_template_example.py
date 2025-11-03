"""
Command-line helper that normalizes subject images and segmentation masks to a
common template using `pyleison_map.preprocess.normalisation.normalize_to_template`.

Example:
    python -m pyleison_map.examples.normalize_to_template_example \
        --image-dir /Users/tonycao/mycode/mutism/imagesTs \
        --mask-dir /Users/tonycao/mycode/mutism/labelsTs \
        --output-dir /tmp/normalized_patients

Requires the `antspyx` package. Install with `pip install antspyx`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from pyleison_map.preprocess.normalisation import normalize_to_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize subject images (and matching masks) to a template."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "templates"
        / "MNI152_template.nii.gz",
        help="Path to the template image in NIfTI format.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing subject images (expects *.nii or *.nii.gz files).",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing subject segmentation masks (matching file names).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory where normalized images and masks will be saved.",
    )
    parser.add_argument(
        "--registration-type",
        default="SyN",
        help="Registration transform family passed to ants.registration (default: SyN).",
    )
    parser.add_argument(
        "--image-interpolator",
        default="lanczosWindowedSinc",
        help="Interpolator used for intensity images (default: lanczosWindowedSinc).",
    )
    parser.add_argument(
        "--mask-interpolator",
        default="nearestNeighbor",
        help="Interpolator used for segmentation masks (default: nearestNeighbor).",
    )
    return parser.parse_args()


def collect_image_pairs(
    image_dir: Path,
    mask_dir: Path,
) -> Tuple[Sequence[str], Sequence[str]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths: List[Path] = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.name.endswith((".nii", ".nii.gz"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No NIfTI images found in {image_dir}")

    mask_paths: List[Path] = []
    missing_masks: List[str] = []

    for image_path in image_paths:
        candidate = mask_dir / image_path.name.replace("_0000", "")
        if candidate.exists():
            mask_paths.append(candidate)
        else:
            missing_masks.append(image_path.name)

    if missing_masks:
        raise FileNotFoundError(
            "Missing masks for: " + ", ".join(sorted(missing_masks))
        )

    return [str(p) for p in image_paths], [str(p) for p in mask_paths]


def main() -> int:
    args = parse_args()

    template_path = args.template
    if not template_path.exists():
        print(f"Template not found at {template_path}", file=sys.stderr)
        return 1

    try:
        images, masks = collect_image_pairs(args.image_dir, args.mask_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    image_output_dir = args.output_dir / "normalized_images"
    mask_output_dir = args.output_dir / "normalized_masks"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = normalize_to_template(
            images=images,
            template=str(template_path),
            segmentations=masks,
            save_registered_images=True,
            image_output_dir=image_output_dir,
            save_registered_segmentations=True,
            segmentation_output_dir=mask_output_dir,
            registration_type=args.registration_type,
            image_interpolator=args.image_interpolator,
            segmentation_interpolator=args.mask_interpolator,
        )
    except ImportError as exc:
        print(
            "Failed to import antspy. Install it with `pip install antspyx` "
            "before running this script.",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    print(
        f"Normalized {len(results)} subjects "
        f"to {template_path.name}. Images: {image_output_dir}, Masks: {mask_output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

