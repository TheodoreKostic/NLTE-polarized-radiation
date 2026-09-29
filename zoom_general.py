"""Create an RGB FITS image and a WCS-aware zoom around a sky position.

The three input FITS files are supplied in red, green, blue order. Each file
must contain a two-dimensional image with celestial WCS metadata. Green and
blue are reprojected onto the red image's pixel grid before being combined.

Example::

    python zoom_general.py red.fits green.fits blue.fits \\
        --target 83.63,-5.39 --output rgb_zoom.png

Target coordinates are ICRS right ascension and declination in degrees.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from reproject import reproject_interp
from regions import (
    CirclePixelRegion,
    EllipsePixelRegion,
    PixelRegion,
    PixCoord,
    Regions,
    SkyRegion,
)


def parse_target(value: str) -> tuple[float, float]:
    """Parse ICRS right ascension and declination in decimal degrees."""
    try:
        coordinates = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target must be two decimal-degree coordinates: RA,DEC"
        ) from error
    if len(coordinates) != 2:
        raise argparse.ArgumentTypeError(
            "target must be two decimal-degree coordinates: RA,DEC"
        )
    ra, dec = coordinates
    if not -90.0 <= dec <= 90.0:
        raise argparse.ArgumentTypeError(
            "target declination must be between -90 and 90 degrees"
        )
    return ra, dec


def read_fits_image(input_path: Path) -> tuple[np.ndarray, WCS]:
    """Read the first 2D FITS image HDU with usable celestial WCS metadata."""
    with fits.open(input_path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=float)
            if data.ndim != 2:
                continue
            wcs = WCS(hdu.header).celestial
            if wcs.has_celestial:
                return data, wcs

    raise ValueError(
        f"{input_path} contains no 2D image HDU with celestial WCS metadata"
    )


def percentile_stretch(channel: np.ndarray) -> np.ndarray:
    """Apply a linear z-scale stretch, ignoring non-finite pixels."""
    finite_values = channel[np.isfinite(channel)]
    if finite_values.size == 0:
        raise ValueError("an RGB channel has no finite pixels after alignment")
    lower, upper = ZScaleInterval().get_limits(finite_values)
    if upper <= lower:
        return np.zeros_like(channel, dtype=float)
    stretched = np.zeros_like(channel, dtype=float)
    valid = np.isfinite(channel)
    stretched[valid] = np.clip(
        (channel[valid] - lower) / (upper - lower), 0.0, 1.0
    )
    return stretched


def zoom_bounds(
    center_x: float,
    center_y: float,
    zoom_size: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Return a square pixel crop, shifted as needed to remain in the image."""
    crop_width = min(zoom_size, width)
    crop_height = min(zoom_size, height)
    left = int(round(center_x)) - crop_width // 2
    top = int(round(center_y)) - crop_height // 2
    left = min(max(left, 0), width - crop_width)
    top = min(max(top, 0), height - crop_height)
    return left, top, left + crop_width, top + crop_height


def read_ds9_regions(regions_path: Path, wcs: WCS) -> list[PixelRegion]:
    """Read DS9 regions and convert sky-coordinate regions to image pixels."""
    contents = regions_path.read_text()
    if re.search(r"^physical\s*$", contents, flags=re.MULTILINE):
        return read_physical_regions(contents)

    pixel_regions = []
    for region in Regions.read(regions_path, format="ds9"):
        if isinstance(region, SkyRegion):
            region = region.to_pixel(wcs)
        if isinstance(region, PixelRegion):
            pixel_regions.append(region)
    return pixel_regions


def read_physical_regions(contents: str) -> list[PixelRegion]:
    """Read DS9 physical-coordinate circles and ellipses as pixel regions."""
    regions = []
    shape_pattern = re.compile(
        r"^(circle|ellipse)\(([^)]*)\)\s*(?:#\s*(.*))?$",
        flags=re.IGNORECASE,
    )
    for line in contents.splitlines():
        match = shape_pattern.match(line.strip())
        if match is None:
            continue
        values = [float(value.strip()) for value in match.group(2).split(",")]
        comment = match.group(3) or ""
        text_match = re.search(r"text=\{([^}]*)\}", comment, flags=re.IGNORECASE)
        color_match = re.search(r"(?:color|edgecolor)=([\w]+)", comment, flags=re.IGNORECASE)
        visual = {"edgecolor": color_match.group(1) if color_match else "green"}
        meta = {"text": text_match.group(1) if text_match else ""}

        if match.group(1).lower() == "circle" and len(values) == 3:
            center = PixCoord(x=values[0] - 1.0, y=values[1] - 1.0)
            regions.append(
                CirclePixelRegion(center, radius=values[2], visual=visual, meta=meta)
            )
        elif match.group(1).lower() == "ellipse" and len(values) == 5:
            center = PixCoord(x=values[0] - 1.0, y=values[1] - 1.0)
            regions.append(
                EllipsePixelRegion(
                    center,
                    width=2 * values[2],
                    height=2 * values[3],
                    angle=Angle(values[4], unit="deg"),
                    visual=visual,
                    meta=meta,
                )
            )
    return regions


def plot_regions(
    regions: list[PixelRegion],
    axis: WCSAxes,
    label_region: PixelRegion | None = None,
) -> None:
    """Draw pixel regions and their optional DS9 text labels on an axis."""
    for region in regions:
        region.plot(ax=axis, origin=(0, 0))
        label = region.meta.get("text")
        if region is label_region:
            label = None
        if label:
            center = region.center
            color = region.visual.get(
                "edgecolor", region.visual.get("color", "yellow")
            )
            axis.text(
                center.x,
                center.y,
                str(label),
                color=color,
                fontsize=7,
                transform=axis.get_transform("pixel"),
                clip_on=True,
            )


def plot_rgb_fits(
    red_path: Path,
    green_path: Path,
    blue_path: Path,
    output_path: Path,
    target_ra: float,
    target_dec: float,
    zoom_size: int,
    target_label: str,
    regions_path: Path | None,
) -> None:
    """Reproject three FITS images to a shared grid and plot a target zoom."""
    red, reference_wcs = read_fits_image(red_path)
    green, green_wcs = read_fits_image(green_path)
    blue, blue_wcs = read_fits_image(blue_path)
    reference_shape = red.shape

    aligned_channels = [red]
    for channel, channel_wcs, channel_path in (
        (green, green_wcs, green_path),
        (blue, blue_wcs, blue_path),
    ):
        aligned, footprint = reproject_interp(
            (channel, channel_wcs),
            reference_wcs,
            shape_out=reference_shape,
        )
        aligned[footprint <= 0] = np.nan
        if not np.isfinite(aligned).any():
            raise ValueError(
                f"{channel_path} has no coverage on the red FITS image's WCS grid"
            )
        aligned_channels.append(aligned)

    target = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg, frame="icrs")
    target_in_image_frame = target.transform_to(
        wcs_to_celestial_frame(reference_wcs)
    )
    target_x, target_y = reference_wcs.world_to_pixel(target_in_image_frame)
    target_x, target_y = float(target_x), float(target_y)
    height, width = reference_shape
    if not (np.isfinite(target_x) and np.isfinite(target_y)):
        raise ValueError("target coordinates could not be mapped onto the reference image")
    if not (0 <= target_x < width and 0 <= target_y < height):
        raise ValueError(
            "target coordinates fall outside the red/reference FITS image "
            f"({width} x {height} pixels; target pixel is "
            f"{target_x:.2f}, {target_y:.2f})"
        )

    left, top, right, bottom = zoom_bounds(
        target_x, target_y, zoom_size, width, height
    )
    rgb = np.stack(
        [percentile_stretch(channel) for channel in aligned_channels], axis=-1
    )
    coverage = np.any(
        np.stack([np.isfinite(channel) for channel in aligned_channels], axis=-1),
        axis=-1,
    )
    rgb[~coverage] = 1.0
    zoom_rgb = rgb[top:bottom, left:right]
    regions = read_ds9_regions(regions_path, reference_wcs) if regions_path else []
    target_pixel = PixCoord(x=target_x, y=target_y)
    target_region = next(
        (region for region in regions if region.contains(target_pixel)),
        None,
    )
    region_label = target_region.meta.get("text") if target_region else None
    displayed_target_label = str(region_label or target_label)

    figure = plt.figure(figsize=(12, 10), constrained_layout=True, facecolor="white")
    axis = figure.add_subplot(111, projection=reference_wcs)
    axis.set_facecolor("white")
    pixel_transform = axis.get_transform("pixel")
    axis.imshow(rgb, origin="lower", interpolation="nearest")
    axis.set_title("RGB field of view")
    axis.coords[0].set_axislabel("Right Ascension")
    axis.coords[1].set_axislabel("Declination")
    axis.plot(
        target_x,
        target_y,
        marker="+",
        color="yellow",
        markersize=14,
        markeredgewidth=2,
        transform=pixel_transform,
    )
    axis.text(
        target_x + 20,
        target_y + 20,
        displayed_target_label,
        color="yellow",
        fontsize=9,
        transform=pixel_transform,
        clip_on=True,
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
        zorder=20,
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
    plot_regions(regions, axis, label_region=target_region)

    inset = inset_axes(
        axis,
        width="38%",
        height="38%",
        loc="upper right",
        borderpad=1.2,
        axes_class=WCSAxes,
        axes_kwargs={"wcs": reference_wcs},
    )
    inset.imshow(
        zoom_rgb,
        origin="lower",
        interpolation="lanczos",
        extent=(left, right, top, bottom),
    )
    inset_pixel_transform = inset.get_transform("pixel")
    inset.set_facecolor("white")
    inset.plot(
        target_x,
        target_y,
        marker="+",
        color="yellow",
        markersize=16,
        markeredgewidth=2,
        transform=inset_pixel_transform,
    )
    inset.text(
        target_x + 20,
        target_y + 20,
        displayed_target_label,
        color="yellow",
        fontsize=9,
        transform=inset_pixel_transform,
        clip_on=True,
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
        zorder=20,
    )
    inset.text(
        0.02,
        0.98,
        f"{displayed_target_label} zoom",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        zorder=20,
    )
    inset.coords[0].set_axislabel("")
    inset.coords[1].set_axislabel("")
    inset.set_xlim(left, right)
    inset.set_ylim(top, bottom)
    inset.tick_params(labelsize=8)
    inset.grid(color="white", alpha=0.3)
    plot_regions(regions, inset, label_region=target_region)

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
    print(f"Target (ICRS): RA={target_ra:.6f} deg, Dec={target_dec:.6f} deg")
    print(f"Target pixel on red/reference grid: ({target_x:.2f}, {target_y:.2f})")
    print(f"Saved RGB field and target zoom to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine three FITS images as red, green, and blue channels, "
            "reproject them to a common grid, and create a target zoom."
        )
    )
    parser.add_argument("red", type=Path, help="red-channel FITS file")
    parser.add_argument("green", type=Path, help="green-channel FITS file")
    parser.add_argument("blue", type=Path, help="blue-channel FITS file")
    parser.add_argument(
        "--target",
        type=parse_target,
        required=True,
        metavar="RA,DEC",
        help="target ICRS right ascension and declination in decimal degrees",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("rgb_zoom.png"),
        help="output image path (default: rgb_zoom.png)",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=500,
        help="width and height of the zoom inset in pixels (default: 500)",
    )
    parser.add_argument(
        "--label",
        default="Target",
        help="target name shown in the inset title (default: Target)",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=None,
        help="optional DS9 .reg file to overlay on the RGB field and zoom",
    )
    args = parser.parse_args()

    if args.zoom_size <= 0:
        parser.error("zoom-size must be positive")

    try:
        plot_rgb_fits(
            args.red,
            args.green,
            args.blue,
            args.output,
            args.target[0],
            args.target[1],
            args.zoom_size,
            args.label,
            args.regions,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()