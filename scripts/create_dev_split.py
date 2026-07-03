"""Create a deterministic labeled train/dev split."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.evaluate import VALID_LABELS, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic train/dev split.")
    parser.add_argument("--train", default="Dataset/train.csv")
    parser.add_argument("--dev-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, dev_path = create_dev_split(
        args.train,
        dev_size=args.dev_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Wrote train split to {train_path}")
    print(f"Wrote dev split to {dev_path}")


def create_dev_split(
    train_path: str | Path,
    *,
    dev_size: float = 0.2,
    seed: int = 42,
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    if not 0 < float(dev_size) < 1:
        raise ValueError("dev_size must be between 0 and 1")

    source_path = Path(train_path)
    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source_path}: CSV file is missing a header row")
        fieldnames = list(reader.fieldnames)
        id_column = _resolve_column(fieldnames, ("id", "example_id", "story_id"))
        label_column = _resolve_column(fieldnames, ("label",))
        if id_column is None:
            raise ValueError(f"{source_path}: missing required id column")
        if label_column is None:
            raise ValueError(f"{source_path}: missing required label column")
        rows = list(reader)

    _validate_rows(source_path, rows, id_column, label_column)
    train_rows, dev_rows = split_rows(rows, id_column=id_column, label_column=label_column, dev_size=dev_size, seed=seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_out = output / "train_split.csv"
    dev_out = output / "dev_split.csv"
    _write_csv(train_out, fieldnames, train_rows)
    _write_csv(dev_out, fieldnames, dev_rows)
    return train_out, dev_out


def split_rows(
    rows: list[dict[str, str]],
    *,
    id_column: str,
    label_column: str,
    dev_size: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[normalize_label(row[label_column], kind="gold")].append(row)

    can_stratify = len(by_label) > 1 and all(len(group) >= 2 for group in by_label.values())
    dev_ids: set[str] = set()
    if can_stratify:
        for label in VALID_LABELS:
            group = list(by_label.get(label, []))
            if not group:
                continue
            rng.shuffle(group)
            dev_count = max(1, round(len(group) * dev_size))
            dev_count = min(dev_count, len(group) - 1)
            dev_ids.update(str(row[id_column]).strip() for row in group[:dev_count])
    else:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        dev_count = max(1, round(len(shuffled) * dev_size))
        dev_count = min(dev_count, len(shuffled) - 1)
        dev_ids.update(str(row[id_column]).strip() for row in shuffled[:dev_count])

    train_rows = [row for row in rows if str(row[id_column]).strip() not in dev_ids]
    dev_rows = [row for row in rows if str(row[id_column]).strip() in dev_ids]
    return train_rows, dev_rows


def _validate_rows(path: Path, rows: list[dict[str, str]], id_column: str, label_column: str) -> None:
    if not rows:
        raise ValueError(f"{path}: no rows to split")
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        example_id = str(row.get(id_column, "")).strip()
        if not example_id:
            raise ValueError(f"{path}: row {row_number} has an empty id")
        if example_id in seen:
            raise ValueError(f"{path}: duplicate id `{example_id}` at row {row_number}")
        seen.add(example_id)
        normalize_label(row.get(label_column), kind="gold")


def _resolve_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {field.strip().lstrip("\ufeff").casefold(): field for field in fieldnames}
    for candidate in candidates:
        actual = lookup.get(candidate.casefold())
        if actual is not None:
            return actual
    return None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
