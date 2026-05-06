from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _load_json(path: Path) -> Dict:
    with path.open() as file:
        return json.load(file)


def _iter_frame_files(frames_root: Path) -> List[Path]:
    files: List[Path] = []
    for path in frames_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return sorted(files)


def _read_png_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _read_jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as file:
        data = file.read()

    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None

    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2

        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break

        segment_length = struct.unpack(">H", data[index:index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            break

        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if index + 7 > len(data):
                break
            height, width = struct.unpack(">HH", data[index + 3:index + 7])
            return int(width), int(height)

        index += segment_length

    return None


def _read_webp_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as file:
        header = file.read(64)

    if len(header) < 30 or header[:4] != b"RIFF" or header[8:12] != b"WEBP":
        return None

    chunk = header[12:16]
    if chunk == b"VP8 " and len(header) >= 30:
        width, height = struct.unpack("<HH", header[26:30])
        return int(width & 0x3FFF), int(height & 0x3FFF)
    if chunk == b"VP8L" and len(header) >= 25:
        bits = struct.unpack("<I", header[21:25])[0]
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return int(width), int(height)
    if chunk == b"VP8X" and len(header) >= 30:
        width = 1 + int.from_bytes(header[24:27], "little")
        height = 1 + int.from_bytes(header[27:30], "little")
        return width, height

    return None


def _read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _read_png_size(path)
    if suffix in {".jpg", ".jpeg"}:
        return _read_jpeg_size(path)
    if suffix == ".webp":
        return _read_webp_size(path)
    return None


def reconcile_dataset(
    annotations_path: Path,
    frames_root: Path,
    output_path: Path,
) -> Dict[str, int]:
    data = _load_json(annotations_path)
    existing_frame_paths = _iter_frame_files(frames_root)
    existing_rel_paths = {
        path.relative_to(frames_root).as_posix(): path for path in existing_frame_paths
    }

    source_images = data.get("images", [])
    source_annotations = data.get("annotations", [])

    annotations_by_image_id: Dict[int, List[Dict]] = {}
    for annotation in source_annotations:
        image_id = int(annotation["image_id"])
        annotations_by_image_id.setdefault(image_id, []).append(annotation)

    kept_images: List[Dict] = []
    kept_annotations: List[Dict] = []
    used_paths = set()
    next_image_id = (
        max((int(image["id"]) for image in source_images), default=0) + 1
    )

    for image in source_images:
        file_name = image["file_name"]
        if file_name not in existing_rel_paths:
            continue
        kept_images.append(image)
        used_paths.add(file_name)
        kept_annotations.extend(annotations_by_image_id.get(int(image["id"]), []))

    added_unannotated_images = 0
    for file_name, full_path in sorted(existing_rel_paths.items()):
        if file_name in used_paths:
            continue

        size = _read_image_size(full_path)
        width, height = size if size is not None else (0, 0)
        kept_images.append(
            {
                "id": next_image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )
        next_image_id += 1
        added_unannotated_images += 1

    reconciled = {
        "licenses": data.get("licenses", []),
        "info": data.get("info", {}),
        "categories": data.get("categories", []),
        "images": sorted(kept_images, key=lambda item: item["file_name"]),
        "annotations": sorted(kept_annotations, key=lambda item: int(item["id"])),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(reconciled, file, separators=(",", ":"))

    return {
        "source_images": len(source_images),
        "source_annotations": len(source_annotations),
        "existing_frames": len(existing_rel_paths),
        "matched_images": len(used_paths),
        "added_unannotated_images": added_unannotated_images,
        "kept_images": len(reconciled["images"]),
        "kept_annotations": len(reconciled["annotations"]),
        "dropped_missing_images": len(source_images) - len(used_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reconciled COCO annotation file aligned to the real frame files."
    )
    parser.add_argument(
        "--annotations",
        default="/Users/kappasutra/MT7/annotations/instances_default.json",
        help="Path to the source COCO annotations JSON file.",
    )
    parser.add_argument(
        "--frames-root",
        default="/Users/kappasutra/MT7/FRAMED-FINAL-INSALLAH",
        help="Root directory containing the actual image frames.",
    )
    parser.add_argument(
        "--output",
        default="/Users/kappasutra/MT7/annotations/instances_reconciled.json",
        help="Path to write the reconciled COCO annotations JSON file.",
    )
    args = parser.parse_args()

    summary = reconcile_dataset(
        annotations_path=Path(args.annotations),
        frames_root=Path(args.frames_root),
        output_path=Path(args.output),
    )

    print("Reconciliation summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
