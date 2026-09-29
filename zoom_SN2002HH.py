"""Create a high-resolution zoomed plot around SN-2002HH.

Example:
    python zoom_SN2002HH.py path/to/image.png

The default crop is tuned for the supplied 587 x 656 pixel image. Use
--crop to provide pixel coordinates as: left top right bottom.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import ConnectionPatch
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


DEFAULT_CROP = (240, 320, 390, 440)
TARGET_PIXEL = (2400.18, 1735.85)
TARGET_REGION_TEXT = "SN-2002HH"


def parse_crop(value: str) -> tuple[int, int, int, int]:
    """Parse a crop supplied as left,top,right,bottom."""
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "crop must contain four integers: left,top,right,bottom"
        ) from error

    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError(
            "crop must contain four integers: left,top,right,bottom"
        )

    left, top, right, bottom = coordinates
    if left >= right or top >= bottom:
        raise argparse.ArgumentTypeError(
            "crop must satisfy left < right and top < bottom"
        )
    return coordinates


def percentile_stretch(channel: np.ndarray) -> np.ndarray:
    """Convert one FITS channel using a DS9-like linear z-scale."""
    finite_values = channel[np.isfinite(channel)]
    lower, upper = ZScaleInterval().get_limits(finite_values)
    if upper <= lower:
        return np.zeros_like(channel, dtype=float)
    return np.clip((channel - lower) / (upper - lower), 0.0, 1.0)


def read_ds9_regions(region_path: Path) -> list[dict[str, object]]:
    """Read physical-coordinate circles from a DS9 region file."""
    circle_pattern = re.compile(
        r"^circle\(([-+\d.eE]+),([-+\d.eE]+),([-+\d.eE]+)\)"
    )
    regions = []
    for line in region_path.read_text().splitlines():
        match = circle_pattern.match(line.strip())
        if match is None:
            continue
        comment = line.partition("#")[2]
        text_match = re.search(r"text=\{([^}]*)\}", comment)
        color_match = re.search(r"color=([\w]+)", comment)
        regions.append(
            {
                "x": float(match.group(1)) - 1.0,
                "y": float(match.group(2)) - 1.0,
                "radius": float(match.group(3)),
                "text": text_match.group(1) if text_match else "",
                "color": color_match.group(1) if color_match else "green",
            }
        )
    return regions


def plot_composite_fits(
    input_path: Path,
    output_path: Path,
    zoom_size: int,
    regions_path: Path,
) -> None:
    """Plot the complete RGB field with a sky-coordinate zoom inset."""
    with fits.open(input_path) as hdul:
        channels = [np.asarray(hdul[index].data, dtype=float) for index in (1, 2, 3)]
        wcs = WCS(hdul[1].header).celestial

    if any(channel.ndim != 2 for channel in channels):
        raise ValueError("composite.fits HDUs 1-3 must each contain a 2D image")
    if len({channel.shape for channel in channels}) != 1:
        raise ValueError("composite.fits RGB channels must have identical shapes")

    target_x, target_y = TARGET_PIXEL
    regions = read_ds9_regions(regions_path)
    target_region = next(
        (region for region in regions if TARGET_REGION_TEXT in str(region["text"])),
        None,
    )
    if target_region is None:
        companion_path = regions_path.with_name("SN2002HHf_regions.reg")
        if companion_path.exists() and companion_path != regions_path:
            companion_regions = read_ds9_regions(companion_path)
            target_region = next(
                (region for region in companion_regions if TARGET_REGION_TEXT in str(region["text"])),
                None,
            )
            regions.extend(companion_regions)
            if target_region is not None:
                print(f"SN-2002HH region found in companion file: {companion_path}")
    if target_region is None:
        raise ValueError(
            f"No region labelled {TARGET_REGION_TEXT!r} found in {regions_path} "
            "or SN2002HHf_regions.reg"
        )

    region_x = float(target_region["x"])
    region_y = float(target_region["y"])
    region_radius = float(target_region["radius"])
    height, width = channels[0].shape
    half_size = zoom_size // 2
    left = max(0, int(round(region_x)) - half_size)
    right = min(width, left + zoom_size)
    top = max(0, int(round(region_y)) - half_size)
    bottom = min(height, top + zoom_size)
    left = max(0, right - zoom_size)
    top = max(0, bottom - zoom_size)

    rgb = np.stack([percentile_stretch(channel) for channel in channels], axis=-1)
    zoom_rgb = rgb[top:bottom, left:right]

    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    axis = figure.add_subplot(111, projection=wcs)
    axis.imshow(rgb, origin="lower", interpolation="nearest")
    axis.set_title("SN-2002HH field of view")
    axis.coords[0].set_axislabel("RA (J2000)")
    axis.coords[1].set_axislabel("DEC (J2000)")
    pixel_transform = axis.get_transform("pixel")
    axis.plot(
        target_x,
        target_y,
        marker="+",
        color="yellow",
        markersize=14,
        markeredgewidth=2,
        transform=pixel_transform,
    )
    for region in regions:
        if region is target_region:
            continue
        axis.add_patch(
            plt.Circle(
                (float(region["x"]), float(region["y"])),
                float(region["radius"]),
                fill=False,
                edgecolor=str(region["color"]),
                linewidth=0.8,
                alpha=0.8,
                transform=pixel_transform,
            )
        )
    axis.add_patch(
        plt.Circle(
            (region_x, region_y),
            region_radius,
            fill=False,
            edgecolor="blue",
            linewidth=1.8,
            transform=pixel_transform,
        )
    )
    for region in regions:
        axis.text(
            float(region["x"]) + float(region["radius"]),
            float(region["y"]) + float(region["radius"]),
            str(region["text"]),
            color=str(region["color"]),
            fontsize=7,
            transform=pixel_transform,
            clip_on=True,
        )
    axis.add_patch(
        plt.Rectangle(
            (left, top),
            right - left,
            bottom - top,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
            transform=pixel_transform,
        )
    )

    inset = inset_axes(
        axis,
        width="38%",
        height="38%",
        loc="upper right",
        borderpad=1.2,
        axes_class=WCSAxes,
        axes_kwargs={"wcs": wcs},
    )
    inset.imshow(
        zoom_rgb,
        origin="lower",
        interpolation="lanczos",
        extent=(left, right, top, bottom),
    )
    inset_pixel_transform = inset.get_transform("pixel")
    inset.plot(target_x, target_y, marker="+", color="yellow", markersize=16, markeredgewidth=2, transform=inset_pixel_transform)
    inset.add_patch(
        plt.Circle(
            (region_x, region_y),
            region_radius,
            fill=False,
            edgecolor="blue",
            linewidth=1.8,
            transform=inset_pixel_transform,
        )
    )
    for region in regions:
        if region is not target_region:
            continue
        region_label_x = float(region["x"]) + float(region["radius"])
        region_label_y = float(region["y"]) + float(region["radius"])
        if left <= float(region["x"]) <= right and top <= float(region["y"]) <= bottom:
            inset.text(
                region_label_x,
                region_label_y,
                str(region["text"]),
                color=str(region["color"]),
                fontsize=8,
                transform=inset_pixel_transform,
                clip_on=True,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
            )
    inset.set_title("SN-2002HH zoom", fontsize=11)
    inset.coords[0].set_axislabel("")
    inset.coords[1].set_axislabel("")
    inset.set_xlim(left, right)
    inset.set_ylim(top, bottom)
    inset.tick_params(labelsize=8)
    inset.grid(color="white", alpha=0.3)

    figure.add_artist(
        ConnectionPatch(
            xyA=(left, bottom),
            coordsA=axis.transData,
            xyB=(0, 0),
            coordsB=inset.transAxes,
            color="cyan",
            linewidth=1.0,
            alpha=0.8,
        )
    )
    figure.add_artist(
        ConnectionPatch(
            xyA=(right, bottom),
            coordsA=axis.transData,
            xyB=(1, 0),
            coordsB=inset.transAxes,
            color="cyan",
            linewidth=1.0,
            alpha=0.8,
        )
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Target pixel: ({target_x:.2f}, {target_y:.2f})")
    print(f"SN-2002HH region center: ({region_x:.2f}, {region_y:.2f})")
    print(f"Saved FITS field and zoom figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop and enlarge the SN-2002HH region of an image."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="input PNG or JPG image",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("SN-2002HH_zoom.png"),
        help="output image path (default: SN-2002HH_zoom.png)",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=DEFAULT_CROP,
        metavar="L,T,R,B",
        help=(
            "crop in pixels: left,top,right,bottom "
            f"(default: {','.join(map(str, DEFAULT_CROP))})"
        ),
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=5.0,
        help="display enlargement factor (default: 5)",
    )
    parser.add_argument(
        "--fits",
        action="store_true",
        help="plot a three-plane RGB FITS file with a sky-coordinate inset",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=500,
        help="width and height of the FITS inset in pixels (default: 500)",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=None,
        help="DS9 region file (default: SN2002HH_regions.reg beside the FITS file)",
    )
    args = parser.parse_args()

    if args.zoom <= 0:
        parser.error("zoom must be positive")
    if args.zoom_size <= 0:
        parser.error("zoom-size must be positive")

    if args.fits or args.image.suffix.lower() in {".fits", ".fit", ".fts"}:
        try:
            regions_path = args.regions or args.image.with_name("SN2002HH_regions.reg")
            if not regions_path.exists():
                parser.error(f"region file does not exist: {regions_path}")
            plot_composite_fits(
                args.image,
                args.output,
                args.zoom_size,
                regions_path,
            )
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
        return

    with Image.open(args.image) as image:
        image = image.convert("RGB")
        left, top, right, bottom = args.crop
        image_width, image_height = image.size
        if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
            parser.error(
                f"crop must fit inside the image dimensions "
                f"({image_width} x {image_height})"
            )
        zoomed = image.crop(args.crop)

    figure_width = max(6.0, zoomed.width * args.zoom / 100.0)
    figure_height = max(4.0, zoomed.height * args.zoom / 100.0)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.imshow(zoomed, interpolation="nearest")
    axis.set_title("SN-2002HH")
    axis.set_xlabel("Pixel x within cropped region")
    axis.set_ylabel("Pixel y within cropped region")
    axis.set_xlim(0, zoomed.width)
    axis.set_ylim(zoomed.height, 0)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved zoomed image to {args.output}")


if __name__ == "__main__":
    main()
