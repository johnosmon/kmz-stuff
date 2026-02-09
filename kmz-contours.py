#!/usr/bin/env python3
"""
kmz-contours.py

Generate distance-to-source isolines (contours) and optionally filled distance bands
from an input KML or KMZ containing KML LineStrings and/or Points.

I/O behavior (modeled after "normalize-geometry.py"):
- Input defaults to STDIN if not provided
- Output defaults to STDOUT if not provided
- Use -i/--input-file and -o/--output-file to specify files
- Automatically detects KML vs KMZ on input (by extension or ZIP signature)
- Defaults to KML output unless -z/--kmz-output is specified
- Silent on success; use -v/--verbose for summary

Output contains ONLY generated geometry (no input points/lines).

Defaults:
  - isolines only
  - if --filled is set: filled polygons + isolines (unless --no-iso)

Dependencies:
  pip install numpy shapely pyproj rasterio scipy matplotlib
"""

import argparse
import io
import math
import os
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import transform as shp_transform

from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio import features
from scipy.ndimage import distance_transform_edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


# ----------------------------
# Input detection / reading
# ----------------------------

def _is_zip_bytes(b: bytes) -> bool:
    # ZIP local file header: PK\x03\x04
    return len(b) >= 4 and b[0:4] == b"PK\x03\x04"


def _read_all_stdin_bytes() -> bytes:
    # Support both text and binary stdin
    if hasattr(sys.stdin, "buffer"):
        return sys.stdin.buffer.read()
    return sys.stdin.read().encode("utf-8")


def _read_input_bytes(input_file: str | None) -> tuple[bytes, str | None]:
    """
    Return (data_bytes, inferred_name).
    inferred_name is a filename-like string if known (used for extension heuristics).
    """
    if input_file:
        p = Path(input_file)
        return p.read_bytes(), p.name
    # stdin
    return _read_all_stdin_bytes(), None


def _detect_input_kind(data: bytes, name: str | None) -> str:
    """
    Return "kmz" or "kml".
    Heuristics:
      - if name endswith .kmz -> kmz
      - if name endswith .kml -> kml
      - else if bytes look like zip -> kmz
      - else -> kml
    """
    if name:
        n = name.lower()
        if n.endswith(".kmz"):
            return "kmz"
        if n.endswith(".kml"):
            return "kml"
    if _is_zip_bytes(data):
        return "kmz"
    return "kml"


def _extract_first_kml_from_kmz_bytes(kmz_bytes: bytes) -> bytes:
    """
    Extract and return bytes for the first .kml found in the KMZ.
    Prefer doc.kml if present.
    """
    with zipfile.ZipFile(io.BytesIO(kmz_bytes), "r") as z:
        names = z.namelist()
        # Prefer doc.kml
        if "doc.kml" in names:
            return z.read("doc.kml")
        # Otherwise first .kml
        kmls = [n for n in names if n.lower().endswith(".kml")]
        if not kmls:
            raise RuntimeError("No .kml found inside KMZ input.")
        return z.read(kmls[0])


# ----------------------------
# KML parsing
# ----------------------------

def _parse_kml_namespace(root):
    ns = {}
    m = re.match(r"\{(.+)\}", root.tag)
    if m:
        ns["kml"] = m.group(1)
    return ns


def _parse_coords_text(coord_text: str):
    """Parse KML <coordinates> block into [(lon, lat), ...]."""
    pts = []
    for token in coord_text.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def load_kml_geometries_from_bytes(kml_bytes: bytes):
    """Return (lines_wgs84, points_wgs84) from KML bytes."""
    root = ET.fromstring(kml_bytes)
    ns = _parse_kml_namespace(root)

    if ns:
        ls_elems = root.findall(".//kml:LineString/kml:coordinates", namespaces=ns)
        pt_elems = root.findall(".//kml:Point/kml:coordinates", namespaces=ns)
    else:
        ls_elems = root.findall(".//LineString/coordinates")
        pt_elems = root.findall(".//Point/coordinates")

    lines = []
    for ce in ls_elems:
        if ce.text and ce.text.strip():
            coords = _parse_coords_text(ce.text.strip())
            if len(coords) >= 2:
                lines.append(LineString(coords))

    points = []
    for pe in pt_elems:
        if pe.text and pe.text.strip():
            coords = _parse_coords_text(pe.text.strip())
            if coords:
                lon, lat = coords[0]
                points.append(Point(lon, lat))

    return lines, points


# ----------------------------
# Distance raster
# ----------------------------

def build_distance_raster(
    lines_wgs84,
    points_wgs84,
    pixel_m: float,
    crs_out: str,
    extent_lonlat: tuple,
    pad_m: float,
):
    """
    Rasterize sources and return:
      dist_m, transform, width, height, to_wgs_transformer
    """
    lon_min, lon_max, lat_min, lat_max = extent_lonlat

    to_proj = Transformer.from_crs("EPSG:4326", crs_out, always_xy=True)
    to_wgs = Transformer.from_crs(crs_out, "EPSG:4326", always_xy=True)

    lines_proj = [shp_transform(lambda x, y: to_proj.transform(x, y), ln) for ln in lines_wgs84]
    pts_proj = [shp_transform(lambda x, y: to_proj.transform(x, y), pt) for pt in points_wgs84]

    x0, y0 = to_proj.transform(lon_min, lat_min)
    x1, y1 = to_proj.transform(lon_max, lat_max)
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)

    xmin -= pad_m
    xmax += pad_m
    ymin -= pad_m
    ymax += pad_m

    width = int(math.ceil((xmax - xmin) / pixel_m))
    height = int(math.ceil((ymax - ymin) / pixel_m))
    transform = from_origin(xmin, ymax, pixel_m, pixel_m)

    shapes = [(g, 1) for g in lines_proj] + [(p, 1) for p in pts_proj]
    if not shapes:
        raise RuntimeError("No LineString or Point geometries found in input.")

    src = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )

    dist_m = distance_transform_edt(src == 0) * pixel_m
    return dist_m, transform, width, height, to_wgs


# ----------------------------
# Geometry helpers
# ----------------------------

def _chaikin_smooth_ring(coords_xy: np.ndarray, iters: int) -> np.ndarray:
    """Chaikin smoothing for a closed ring."""
    if iters <= 0:
        return coords_xy

    pts = np.asarray(coords_xy, dtype=float)
    if pts.shape[0] < 4:
        return pts

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    for _ in range(iters):
        new_pts = []
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)
        pts = np.vstack(new_pts)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

    return pts


def _clean_polygon(poly: Polygon, simplify_m: float, chaikin_iters: int, min_area_m2: float) -> Polygon | None:
    """Optional smoothing/simplify/repair for filled-band polygons."""
    if poly.is_empty:
        return None

    if chaikin_iters > 0:
        ext = np.asarray(poly.exterior.coords)
        ext_s = _chaikin_smooth_ring(ext, chaikin_iters)
        holes_s = []
        for ring in poly.interiors:
            rr = np.asarray(ring.coords)
            holes_s.append(_chaikin_smooth_ring(rr, chaikin_iters))
        try:
            poly = Polygon(ext_s, holes=[h.tolist() for h in holes_s if len(h) >= 4])
        except Exception:
            pass

    if simplify_m > 0:
        poly = poly.simplify(simplify_m, preserve_topology=True)

    poly = poly.buffer(0)

    if poly.is_empty:
        return None
    if min_area_m2 > 0 and poly.area < min_area_m2:
        return None

    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)

    return poly


# ----------------------------
# KML generation
# ----------------------------

def _kml_color_aabbggrr(r: int, g: int, b: int, a: int) -> str:
    """KML expects colors as AABBGGRR."""
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"


def _kml_header(name: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{name}</name>
'''


def _kml_footer() -> str:
    return "</Document></kml>"


def _iso_lines_kml(levels_m, allsegs_xy, to_wgs: Transformer, style_id: str = "isoLine") -> str:
    body = []
    body.append(f"""
<Style id="{style_id}">
  <LineStyle><width>1.5</width></LineStyle>
</Style>
""")
    for lvl_m, segs in zip(levels_m, allsegs_xy):
        if not segs:
            continue
        lvl_km = int(round(lvl_m / 1000.0))
        body.append(f"<Folder><name>Iso {lvl_km} km</name>")
        body.append(f'<Placemark><name>{lvl_km} km</name><styleUrl>#{style_id}</styleUrl><MultiGeometry>')
        for seg in segs:
            seg = np.asarray(seg)
            if seg.shape[0] < 2:
                continue
            lons, lats = to_wgs.transform(seg[:, 0], seg[:, 1])
            coords = " ".join([f"{lon:.6f},{lat:.6f},0" for lon, lat in zip(lons, lats)])
            body.append(f"<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>")
        body.append("</MultiGeometry></Placemark>")
        body.append("</Folder>")
    return "\n".join(body)


def _filled_bands_kml_from_allsegs(
    levels_m: np.ndarray,
    allsegs_xy,
    to_wgs: Transformer,
    cmap_name: str,
    alpha: float,
    simplify_m: float,
    chaikin_iters: int,
    min_area_m2: float,
    outline_width: float,
) -> str:
    """
    Export filled contour bands as polygons using QuadContourSet.allsegs.
    Reconstruct holes by containment.
    """
    n_bands = len(allsegs_xy)
    if n_bands <= 0:
        return ""

    cmap = cm.get_cmap(cmap_name, n_bands)
    a = int(max(0.0, min(1.0, alpha)) * 255)

    styles = []
    placemarks = []

    for i in range(n_bands):
        rgba = cmap(i)
        r = int(round(rgba[0] * 255))
        g = int(round(rgba[1] * 255))
        b = int(round(rgba[2] * 255))
        kml_col = _kml_color_aabbggrr(r, g, b, a)
        styles.append(f"""
<Style id="band{i}">
  <LineStyle><width>{outline_width}</width><color>{kml_col}</color></LineStyle>
  <PolyStyle><color>{kml_col}</color><fill>1</fill><outline>1</outline></PolyStyle>
</Style>
""")

    def rings_to_polys(rings):
        polys = []
        for ring in rings:
            ring = np.asarray(ring)
            if ring.shape[0] < 4:
                continue
            if not np.allclose(ring[0], ring[-1]):
                ring = np.vstack([ring, ring[0]])
            try:
                p = Polygon(ring)
            except Exception:
                continue
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty or p.area <= 0:
                continue
            polys.append(p)

        if not polys:
            return []

        polys.sort(key=lambda p: p.area, reverse=True)

        used_as_hole = set()
        shells = []

        for idx, shell in enumerate(polys):
            if idx in used_as_hole:
                continue
            holes = []
            for jdx in range(idx + 1, len(polys)):
                if jdx in used_as_hole:
                    continue
                cand = polys[jdx]
                if shell.contains(cand.representative_point()):
                    holes.append(cand.exterior.coords[:])
                    used_as_hole.add(jdx)
            try:
                shp = Polygon(shell.exterior.coords[:], holes=holes)
            except Exception:
                shp = shell
            shells.append(shp)

        return shells

    for i in range(n_bands):
        lo = levels_m[i]
        hi = levels_m[i + 1]
        rings = allsegs_xy[i]
        if not rings:
            continue

        polys = rings_to_polys(rings)

        poly_kmls = []
        for poly in polys:
            poly = _clean_polygon(poly, simplify_m=simplify_m, chaikin_iters=chaikin_iters, min_area_m2=min_area_m2)
            if poly is None or poly.is_empty:
                continue

            ext = np.asarray(poly.exterior.coords)
            lons, lats = to_wgs.transform(ext[:, 0], ext[:, 1])
            ext_coords = " ".join([f"{lon:.6f},{lat:.6f},0" for lon, lat in zip(lons, lats)])

            hole_blocks = []
            for ring in poly.interiors:
                rr = np.asarray(ring.coords)
                hlons, hlats = to_wgs.transform(rr[:, 0], rr[:, 1])
                hole_coords = " ".join([f"{lon:.6f},{lat:.6f},0" for lon, lat in zip(hlons, hlats)])
                hole_blocks.append(
                    f"<innerBoundaryIs><LinearRing><coordinates>{hole_coords}</coordinates></LinearRing></innerBoundaryIs>"
                )

            poly_kmls.append(
                f"""
<Polygon>
  <outerBoundaryIs><LinearRing><coordinates>{ext_coords}</coordinates></LinearRing></outerBoundaryIs>
  {''.join(hole_blocks)}
</Polygon>
""".strip()
            )

        if not poly_kmls:
            continue

        lo_km = lo / 1000.0
        hi_km = hi / 1000.0
        name = f"Band {lo_km:g}–{hi_km:g} km"

        placemarks.append(f"""
<Placemark>
  <name>{name}</name>
  <styleUrl>#band{i}</styleUrl>
  <MultiGeometry>
    {''.join(poly_kmls)}
  </MultiGeometry>
</Placemark>
""".strip())

    return "\n".join(styles) + "\n" + "\n".join(placemarks)


def _kml_to_kmz_bytes(kml_text: str) -> bytes:
    """Return KMZ (zip) bytes containing doc.kml."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml_text.encode("utf-8"))
    return bio.getvalue()


def _write_output_bytes(output_file: str | None, data: bytes):
    if output_file:
        Path(output_file).write_bytes(data)
        return
    # stdout
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(data)
    else:
        # last resort
        sys.stdout.write(data.decode("utf-8", errors="replace"))


# ----------------------------
# Argparse
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kmz-contours.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Generate distance-to-source isolines (contours) and optionally filled distance bands\n"
            "from an input KML or KMZ containing KML LineStrings and/or Points.\n\n"
            "I/O (like normalize-geometry.py):\n"
            "  - input defaults to STDIN; use -i/--input-file\n"
            "  - output defaults to STDOUT; use -o/--output-file\n"
            "  - input auto-detects KML vs KMZ\n"
            "  - output defaults to KML; add -z/--kmz-output for KMZ\n\n"
            "Defaults:\n"
            "  - isolines only\n"
            "  - with --filled: filled polygons + isolines (unless --no-iso)\n\n"
            "Silent on success; use -v/--verbose for summary.\n"
        ),
        epilog=(
            "Examples:\n"
            "  KMZ input from file, KML to stdout:\n"
            "    kmz-contours.py -i in.kmz > out.kml\n\n"
            "  KML input from stdin, KMZ output to file:\n"
            "    cat in.kml | kmz-contours.py -z -o out.kmz\n\n"
            "  Filled + isolines (default with --filled), write KML:\n"
            "    kmz-contours.py -i in.kmz -o out.kml --filled --pixel-m 250 --interval-km 5\n"
        ),
    )

    p.add_argument("-i", "--input-file", type=str, default=None,
                   help="Input file (KML or KMZ). If omitted, read from STDIN.")
    p.add_argument("-o", "--output-file", type=str, default=None,
                   help="Output file. If omitted, write to STDOUT.")
    p.add_argument("-z", "--kmz-output", action="store_true",
                   help="Write KMZ output (zip with doc.kml). Default is KML output.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print summary information on success.")

    p.add_argument("--filled", action="store_true",
                   help="Generate filled distance bands (polygons). Isolines are also included by default.")
    p.add_argument("--no-iso", action="store_true",
                   help="When used with --filled, omit isolines from the output.")

    p.add_argument("--pixel-m", type=float, default=1000.0,
                   help="Raster pixel size in meters (default: 1000)")
    p.add_argument("--interval-km", type=float, default=10.0,
                   help="Contour/band interval in km (default: 10)")
    p.add_argument("--max-km", type=float, default=None,
                   help="Maximum distance in km (default: auto from raster)")

    p.add_argument("--crs", type=str, default="EPSG:26913",
                   help="Projected CRS for distance calculations (default: EPSG:26913)")
    p.add_argument("--pad-m", type=float, default=10000.0,
                   help="Padding around analysis extent in meters (default: 10000)")

    # Default extent: New Mexico bbox
    p.add_argument("--lon-min", type=float, default=-109.05, help="Extent min longitude (default: -109.05)")
    p.add_argument("--lon-max", type=float, default=-103.00, help="Extent max longitude (default: -103.00)")
    p.add_argument("--lat-min", type=float, default=31.33, help="Extent min latitude (default: 31.33)")
    p.add_argument("--lat-max", type=float, default=37.00, help="Extent max latitude (default: 37.00)")

    p.add_argument("--debug-png", type=str, default=None,
                   help="Write a debug PNG rendering to this path")

    # Filled-mode knobs
    p.add_argument("--cmap", type=str, default="viridis",
                   help="Matplotlib colormap for filled bands (default: viridis)")
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Fill opacity 0..1 (default: 0.55)")
    p.add_argument("--outline-width", type=float, default=1.2,
                   help="Polygon outline width (default: 1.2)")
    p.add_argument("--simplify-m", type=float, default=0.0,
                   help="Simplify polygons in meters (default: 0 = off)")
    p.add_argument("--chaikin-iters", type=int, default=0,
                   help="Chaikin smoothing iterations (default: 0 = off)")
    p.add_argument("--min-area-m2", type=float, default=0.0,
                   help="Drop polygons smaller than this area (m^2) (default: 0 = off)")

    return p


# ----------------------------
# Main
# ----------------------------

def main():
    args = build_arg_parser().parse_args()

    # Read + detect input
    data, inferred_name = _read_input_bytes(args.input_file)
    kind = _detect_input_kind(data, inferred_name)
    if kind == "kmz":
        kml_bytes = _extract_first_kml_from_kmz_bytes(data)
    else:
        kml_bytes = data

    # Parse geometries
    lines, points = load_kml_geometries_from_bytes(kml_bytes)
    if not lines and not points:
        print("ERROR: No LineString or Point geometries found.", file=sys.stderr)
        sys.exit(2)

    # Distance raster
    extent = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    dist_m, xform, width, height, to_wgs = build_distance_raster(
        lines, points, args.pixel_m, args.crs, extent, args.pad_m
    )

    max_km_auto = float(dist_m.max() / 1000.0)
    max_km = args.max_km if args.max_km is not None else max_km_auto
    interval_km = args.interval_km

    if max_km <= 0 or interval_km <= 0:
        print("ERROR: max-km and interval-km must be positive.", file=sys.stderr)
        sys.exit(2)

    # Levels
    if args.filled:
        top = math.ceil(max_km / interval_km) * interval_km
        band_levels_km = np.arange(0.0, top + interval_km, interval_km, dtype=float)
        if len(band_levels_km) < 2:
            band_levels_km = np.array([0.0, interval_km], dtype=float)
        band_levels_m = band_levels_km * 1000.0
    else:
        band_levels_m = None

    iso_top = math.floor(max_km / interval_km) * interval_km
    iso_levels_km = np.arange(interval_km, iso_top + interval_km, interval_km, dtype=float)
    iso_levels_m = iso_levels_km * 1000.0

    # Meshgrid
    xmin = xform.c
    ymax = xform.f
    pixel = args.pixel_m
    xs = xmin + (np.arange(width) + 0.5) * pixel
    ys = ymax - (np.arange(height) + 0.5) * pixel
    X, Y = np.meshgrid(xs, ys)

    parts = []
    kml_name = "Distance isolines"
    include_iso = (not args.filled) or (args.filled and not args.no_iso)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    if args.filled:
        kml_name = "Distance bands + isolines" if include_iso else "Distance bands"
        cf = ax.contourf(X, Y, dist_m, levels=band_levels_m)

        band_levels_snapshot = np.array(cf.levels, copy=True)
        band_allsegs_snapshot = [[np.array(seg, copy=True) for seg in band] for band in cf.allsegs]

        parts.append(_filled_bands_kml_from_allsegs(
            levels_m=band_levels_snapshot,
            allsegs_xy=band_allsegs_snapshot,
            to_wgs=to_wgs,
            cmap_name=args.cmap,
            alpha=args.alpha,
            simplify_m=args.simplify_m,
            chaikin_iters=args.chaikin_iters,
            min_area_m2=args.min_area_m2,
            outline_width=args.outline_width,
        ))

    if include_iso and len(iso_levels_m) > 0:
        cs = ax.contour(X, Y, dist_m, levels=iso_levels_m)
        iso_levels_snapshot = np.array(cs.levels, copy=True)
        iso_allsegs_snapshot = [[np.array(seg, copy=True) for seg in level_segs] for level_segs in cs.allsegs]
        parts.append(_iso_lines_kml(iso_levels_snapshot, iso_allsegs_snapshot, to_wgs, style_id="isoLine"))

    if args.debug_png:
        ax.set_title(kml_name)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(args.debug_png, dpi=200)

    plt.close(fig)

    kml_text = _kml_header(kml_name) + "\n".join(parts) + "\n" + _kml_footer()

    # Output encoding: default KML; -z => KMZ
    if args.kmz_output:
        out_bytes = _kml_to_kmz_bytes(kml_text)
    else:
        out_bytes = kml_text.encode("utf-8")

    _write_output_bytes(args.output_file, out_bytes)

    if args.verbose:
        in_desc = args.input_file if args.input_file else "STDIN"
        out_desc = args.output_file if args.output_file else "STDOUT"
        out_fmt = "KMZ" if args.kmz_output else "KML"
        mode = "isolines only"
        if args.filled:
            mode = "filled bands + isolines" if include_iso else "filled bands only"

        print(f"Input: {in_desc} ({kind.upper()})", file=sys.stderr)
        print(f"Output: {out_desc} ({out_fmt})", file=sys.stderr)
        print(f"Mode: {mode}", file=sys.stderr)
        print(f"Sources: {len(lines)} LineStrings, {len(points)} Points", file=sys.stderr)
        print(f"Pixel: {args.pixel_m} m  Interval: {args.interval_km} km  MaxDist(auto): {max_km_auto:.2f} km", file=sys.stderr)


if __name__ == "__main__":
    main()
