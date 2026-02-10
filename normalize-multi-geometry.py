#!/usr/bin/env python3
"""
normalize-multi-geometry.py

Normalize Google Earth KML/KMZ linework for QGIS/Google Earth:
- Explode MultiGeometry placemarks into individual LineString placemarks (when >1 LineString)
- Clean all LineStrings:
    * drop non-finite coords
    * remove consecutive duplicate XY vertices
    * drop degenerate LineStrings (<2 vertices)
- Ensure Google Earth–compliant KML root:
    * <kml> must contain exactly one top-level Feature (typically <Document>)
      If <kml> has multiple root features (e.g., multiple <Placemark>), wrap in <Document>.

I/O behavior:
- Default input:  STDIN  (KML or KMZ autodetected)
- Default output: STDOUT as KML (UTF-8 XML)
- Optional -i/--input-file and -o/--output-file
- Use -z/--kmz-out to output KMZ instead of KML

Verbosity:
- Silent success by default
- -v/--verbose prints summaries to STDERR

Exit codes:
- 0: success
- 2: usage / input format errors
- 1: unexpected runtime errors
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------
# KML helpers
# ---------------------------

def ns_uri_from_root(root: ET.Element) -> str:
    m = re.match(r"^\{([^}]+)\}", root.tag)
    return m.group(1) if m else ""


def q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}" if ns else tag


def build_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    parent: dict[ET.Element, ET.Element] = {}
    for p in root.iter():
        for c in list(p):
            parent[c] = p
    return parent


def ensure_single_document_root(kml_root: ET.Element) -> None:
    """
    Google Earth expects exactly one root Feature under <kml>.

    Valid examples:
      <kml><Document>...</Document></kml>
      <kml><Folder>...</Folder></kml>        (still a single Feature)
      <kml><Placemark>...</Placemark></kml>  (still a single Feature)

    Invalid for Google Earth:
      <kml><Placemark/> <Placemark/> ...</kml>

    This function wraps multiple root children into a new <Document>.
    It leaves <kml><Document/></kml> unchanged.
    """
    ns = ns_uri_from_root(kml_root)
    document_tag = q(ns, "Document")

    children = list(kml_root)
    if len(children) == 1 and children[0].tag == document_tag:
        return

    # Wrap all existing children under a new Document
    doc = ET.Element(document_tag)
    for child in children:
        kml_root.remove(child)
        doc.append(child)
    kml_root.append(doc)


# ---------------------------
# Coordinate cleaning
# ---------------------------

Coord = Tuple[float, float, Optional[float]]


def parse_coords(txt: str) -> List[Coord]:
    pts: List[Coord] = []
    for token in txt.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            alt = float(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
        except ValueError:
            continue
        pts.append((lon, lat, alt))
    return pts


def is_finite_pt(p: Coord) -> bool:
    lon, lat, alt = p
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return False
    if alt is not None and not math.isfinite(alt):
        return False
    return True


def same_xy(a: Coord, b: Coord) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def coords_to_text(pts: List[Coord]) -> str:
    out = []
    for lon, lat, alt in pts:
        if alt is None:
            out.append(f"{lon:.8f},{lat:.8f}")
        else:
            out.append(f"{lon:.8f},{lat:.8f},{alt:g}")
    return " ".join(out)


def clean_linestring_pts(pts: List[Coord]) -> List[Coord]:
    # Drop non-finite points
    pts = [p for p in pts if is_finite_pt(p)]
    if not pts:
        return []

    # Remove consecutive duplicate XY vertices
    cleaned = [pts[0]]
    for p in pts[1:]:
        if not same_xy(p, cleaned[-1]):
            cleaned.append(p)

    return cleaned


# ---------------------------
# Input detection / parsing
# ---------------------------

def looks_like_zip(b: bytes) -> bool:
    # ZIP local file header signature
    return len(b) >= 4 and b[:4] == b"PK\x03\x04"


def parse_kml_bytes(kml_bytes: bytes) -> ET.Element:
    # Strip leading whitespace/BOM-ish bytes safely
    kml_bytes = kml_bytes.lstrip()
    return ET.fromstring(kml_bytes)


def read_kmz_choose_kml_from_bytes(kmz_bytes: bytes) -> Tuple[str, bytes, List[Tuple[str, bytes]]]:
    """
    Return (chosen_kml_name, chosen_kml_bytes, other_members).

    - If doc.kml exists, use it.
    - Else choose largest *.kml.
    - Preserve all non-KML members (icons, overlays, etc.)
    """
    kmls: List[Tuple[str, bytes]] = []
    others: List[Tuple[str, bytes]] = []

    with zipfile.ZipFile(io.BytesIO(kmz_bytes), "r") as z:
        for name in z.namelist():
            data = z.read(name)
            if name.lower().endswith(".kml"):
                kmls.append((name, data))
            else:
                others.append((name, data))

    if not kmls:
        raise ValueError("No .kml found inside KMZ")

    for name, data in kmls:
        if name == "doc.kml":
            return name, data, others

    name, data = max(kmls, key=lambda t: len(t[1]))
    return name, data, others


def load_input_as_kml_tree(input_bytes: bytes) -> Tuple[str, ET.Element, List[Tuple[str, bytes]]]:
    """
    Autodetect KMZ vs KML.

    Returns:
      chosen_kml_name: for KMZ, selected embedded KML member; for KML, "(input.kml)"
      root: ElementTree root element (<kml>...)
      others: non-KML KMZ members to preserve on KMZ output (empty for KML input)
    """
    if looks_like_zip(input_bytes):
        chosen_name, chosen_kml, others = read_kmz_choose_kml_from_bytes(input_bytes)
        root = parse_kml_bytes(chosen_kml)
        return chosen_name, root, others

    # Treat as KML XML
    root = parse_kml_bytes(input_bytes)
    return "(input.kml)", root, []


# ---------------------------
# Output writers
# ---------------------------

def write_kml_to_bytes(root: ET.Element) -> bytes:
    ns = ns_uri_from_root(root)
    if ns:
        ET.register_namespace("", ns)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_kmz_to_bytes(doc_kml_bytes: bytes, others: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", doc_kml_bytes)
        for name, data in others:
            z.writestr(name, data)
    return buf.getvalue()


# ---------------------------
# Transform: explode + clean
# ---------------------------

@dataclass
class Summary:
    input_kind: str
    chosen_kml_name: str
    lines_before: int
    placemarks_exploded: int
    exploded_lines_created: int
    lines_after_explode: int
    lines_seen_clean: int
    lines_fixed_clean: int
    lines_removed_clean: int
    lines_after_clean: int


def count_linestrings(root: ET.Element) -> int:
    ns = ns_uri_from_root(root)
    linestring_tag = q(ns, "LineString")
    return sum(1 for _ in root.iter(linestring_tag))


def explode_multigeometry_lines(root: ET.Element) -> Tuple[int, int]:
    """
    Explode placemarks that contain MultiGeometry and >1 LineString into multiple placemarks.
    Returns: (placemarks_exploded, lines_created)
    """
    ns = ns_uri_from_root(root)
    if ns:
        ET.register_namespace("", ns)

    placemark_tag = q(ns, "Placemark")
    multigeom_tag = q(ns, "MultiGeometry")
    linestring_tag = q(ns, "LineString")

    name_tag = q(ns, "name")
    styleurl_tag = q(ns, "styleUrl")
    visibility_tag = q(ns, "visibility")
    extended_tag = q(ns, "ExtendedData")
    desc_tag = q(ns, "description")

    parent = build_parent_map(root)

    pms_exploded = 0
    lines_created = 0

    placemarks = list(root.iter(placemark_tag))
    for pm in placemarks:
        if pm.find(f".//{multigeom_tag}") is None:
            continue

        lines = pm.findall(f".//{linestring_tag}")
        if len(lines) <= 1:
            continue

        p = parent.get(pm)
        if p is None:
            continue

        siblings = list(p)
        try:
            idx = siblings.index(pm)
        except ValueError:
            idx = len(siblings)

        base_name_el = pm.find(name_tag)
        base_name = (base_name_el.text or "").strip() if base_name_el is not None else ""

        style_el = pm.find(styleurl_tag)
        vis_el = pm.find(visibility_tag)
        ext_el = pm.find(extended_tag)
        desc_el = pm.find(desc_tag)

        p.remove(pm)

        for i, line in enumerate(lines, start=1):
            new_pm = ET.Element(placemark_tag)

            nm = ET.SubElement(new_pm, name_tag)
            nm.text = f"{base_name} ({i})" if base_name else f"Line ({i})"

            if style_el is not None:
                new_pm.append(deepcopy(style_el))
            if vis_el is not None:
                new_pm.append(deepcopy(vis_el))
            if desc_el is not None:
                new_pm.append(deepcopy(desc_el))
            if ext_el is not None:
                new_pm.append(deepcopy(ext_el))

            # Attach LineString directly under Placemark
            new_pm.append(deepcopy(line))

            p.insert(idx + (i - 1), new_pm)
            lines_created += 1

        pms_exploded += 1

    return pms_exploded, lines_created


def clean_all_linestrings(root: ET.Element) -> Tuple[int, int, int]:
    """
    Clean LineStrings in-place.
    Returns: (lines_seen, lines_fixed, lines_removed)
    """
    ns = ns_uri_from_root(root)
    if ns:
        ET.register_namespace("", ns)

    linestring_tag = q(ns, "LineString")
    coords_tag = q(ns, "coordinates")

    parent = build_parent_map(root)

    seen = 0
    fixed = 0
    removed = 0

    for ls in list(root.iter(linestring_tag)):
        seen += 1
        coords_el = ls.find(coords_tag)
        if coords_el is None or not (coords_el.text or "").strip():
            continue

        pts = parse_coords(coords_el.text)
        cleaned = clean_linestring_pts(pts)

        if len(cleaned) < 2:
            p = parent.get(ls)
            if p is not None:
                p.remove(ls)
                removed += 1
            continue

        if cleaned != pts:
            coords_el.text = coords_to_text(cleaned)
            fixed += 1

    return seen, fixed, removed


# ---------------------------
# CLI / main
# ---------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="normalize-multi-geometry.py",
        description="Normalize Google Earth KML/KMZ MultiGeometry LineStrings for QGIS/Google Earth (explode + clean).",
        add_help=True,  # provides -h/--help
    )
    p.add_argument("-i", "--input-file", help="Input file (KML or KMZ). Default: read from STDIN.")
    p.add_argument("-o", "--output-file", help="Output file. Default: write to STDOUT.")
    p.add_argument(
        "-z", "--kmz-out",
        action="store_true",
        help="Write KMZ output (doc.kml + preserved assets). Default output is KML."
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print a processing summary to STDERR. Default: silent on success."
    )
    return p.parse_args(argv)


def read_input_bytes(args: argparse.Namespace) -> bytes:
    if args.input_file:
        with open(args.input_file, "rb") as f:
            return f.read()
    return sys.stdin.buffer.read()


def write_output_bytes(args: argparse.Namespace, out_bytes: bytes) -> None:
    if args.output_file:
        with open(args.output_file, "wb") as f:
            f.write(out_bytes)
    else:
        sys.stdout.buffer.write(out_bytes)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        input_bytes = read_input_bytes(args)
        if not input_bytes:
            raise ValueError("Empty input")

        chosen_kml_name, root, others = load_input_as_kml_tree(input_bytes)
        input_kind = "KMZ" if looks_like_zip(input_bytes) else "KML"

        lines_before = count_linestrings(root)

        pms_exploded, lines_created = explode_multigeometry_lines(root)
        lines_after_explode = count_linestrings(root)

        seen, fixed, removed = clean_all_linestrings(root)
        lines_after_clean = count_linestrings(root)

        # Ensure GE-compliant single root feature under <kml>
        ensure_single_document_root(root)

        # Serialize KML
        doc_kml_bytes = write_kml_to_bytes(root)

        # Output selection
        if args.kmz_out:
            out_bytes = write_kmz_to_bytes(doc_kml_bytes, others)
        else:
            out_bytes = doc_kml_bytes

        write_output_bytes(args, out_bytes)

        if args.verbose:
            s = Summary(
                input_kind=input_kind,
                chosen_kml_name=chosen_kml_name,
                lines_before=lines_before,
                placemarks_exploded=pms_exploded,
                exploded_lines_created=lines_created,
                lines_after_explode=lines_after_explode,
                lines_seen_clean=seen,
                lines_fixed_clean=fixed,
                lines_removed_clean=removed,
                lines_after_clean=lines_after_clean,
            )
            print(
                "\n".join([
                    f"[info] input kind: {s.input_kind}",
                    f"[info] chosen KML: {s.chosen_kml_name}",
                    f"[info] LineStrings before: {s.lines_before}",
                    f"[explode] placemarks exploded: {s.placemarks_exploded}",
                    f"[explode] new placemarks created (1 per LineString): {s.exploded_lines_created}",
                    f"[info] LineStrings after explode: {s.lines_after_explode}",
                    f"[clean] LineStrings seen: {s.lines_seen_clean}",
                    f"[clean] LineStrings fixed (dedup/finite): {s.lines_fixed_clean}",
                    f"[clean] LineStrings removed (degenerate): {s.lines_removed_clean}",
                    f"[info] LineStrings after clean: {s.lines_after_clean}",
                    f"[output] format: {'KMZ' if args.kmz_out else 'KML'}",
                ]),
                file=sys.stderr,
            )

        return 0

    except (zipfile.BadZipFile, ET.ParseError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
