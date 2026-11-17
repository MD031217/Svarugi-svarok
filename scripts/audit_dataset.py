"""Read-only audit of the original flat YOLO detection dataset."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.schemas import YoloLabel


def audit_dataset(root: Path) -> dict[str, object]:
    names = (root / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
    if not names or any(not name.strip() for name in names) or len(names) != len(set(names)):
        raise ValueError("classes.txt must contain unique, non-empty names")
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = sorted(p for p in root.glob("*.txt") if p.name != "classes.txt")
    if not images:
        raise ValueError("No images found")
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    dimensions: Counter[str] = Counter()
    prefixes: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    canonical_labels: dict[str, set[str]] = {}
    empty = []
    errors = []
    duplicate_rows = []
    for path in labels:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        if not any(line.strip() for line in lines):
            empty.append(path.name)
        seen: set[str] = set()
        present: set[int] = set()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                label = YoloLabel.from_line(line)
                if label.class_id >= len(names):
                    raise ValueError("Class ID is absent from classes.txt")
            except ValueError as exc:
                errors.append({"file": path.name, "line": number, "error": str(exc)})
                continue
            canonical = label.to_line()
            if canonical in seen:
                duplicate_rows.append({"file": path.name, "line": number})
            seen.add(canonical)
            counts[label.class_id] += 1
            present.add(label.class_id)
        image_counts.update(present)
        canonical_labels[path.stem] = seen
    for path in images:
        prefixes[path.stem.split(" (", 1)[0]] += 1
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                image.verify()
            # JPEG verify() alone does not decode pixel data.
            with Image.open(path) as image:
                image.load()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    conflicting_groups = [
        group
        for group in duplicate_groups
        if any(
            canonical_labels.get(Path(name).stem) != canonical_labels.get(Path(group[0]).stem)
            for name in group[1:]
        )
    ]
    return {
        "image_count": len(images),
        "label_file_count": len(labels),
        "object_count": sum(counts.values()),
        "classes": [
            {"id": i, "name": name, "objects": counts[i], "images": image_counts[i]}
            for i, name in enumerate(names)
        ],
        "empty_label_count": len(empty),
        "empty_labels": empty,
        "missing_labels": sorted(image_stems - label_stems),
        "orphan_labels": sorted(label_stems - image_stems),
        "duplicate_image_stems": [s for s, n in Counter(p.stem for p in images).items() if n > 1],
        "duplicate_image_groups": duplicate_groups,
        "conflicting_duplicate_labels": conflicting_groups,
        "duplicate_label_rows": duplicate_rows,
        "image_dimensions": dict(sorted(dimensions.items())),
        "filename_prefixes": dict(sorted(prefixes.items())),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    raise SystemExit(
        int(
            any(
                report[key]
                for key in ("errors", "missing_labels", "orphan_labels", "duplicate_image_stems")
            )
        )
    )


if __name__ == "__main__":
    main()
