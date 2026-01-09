#!/usr/bin/env python3
"""Evaluate classification accuracy on a folder dataset.

Dataset layout (required):
  data/test/<label>/*.jpg|*.jpeg|*.png

- Uses folder name as ground-truth label.
- Loads the existing model + transform from app.py (no code changes).
- Computes Top-1 and Top-3 accuracy.

Output is designed to be reproducible and report-friendly.
Exit code is non-zero if the dataset is missing or no images are found.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class EvalResult:
    total: int
    top1_correct: int
    top3_correct: int
    elapsed_s: float


def _iter_images(root: Path):
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
                yield label, path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "test"

    if not data_root.exists():
        print(f"FAIL: dataset folder not found: {data_root}")
        print("Expected layout: data/test/<label>/*.jpg|*.jpeg|*.png")
        return 2

    # app.py loads classes.txt/model.pth using relative paths at import time.
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    start_import = time.perf_counter()
    try:
        import importlib

        sys.modules.pop("app", None)
        app_module = importlib.import_module("app")
    except Exception as exc:
        elapsed = time.perf_counter() - start_import
        print(f"FAIL: importing app raised {type(exc).__name__}: {exc}")
        print(f"Import elapsed: {elapsed:.3f}s")
        return 3

    import_elapsed = time.perf_counter() - start_import

    try:
        import torch
        from PIL import Image

        model = app_module.model
        class_names = list(app_module.class_names)
        transform = app_module.transform

        # Map ground-truth folder names to model class labels.
        class_set = set(class_names)

        total = 0
        top1_correct = 0
        top3_correct = 0

        start_eval = time.perf_counter()

        with torch.no_grad():
            for gt_label, img_path in _iter_images(data_root):
                total += 1

                if gt_label not in class_set:
                    # Skip unknown labels rather than silently counting as wrong.
                    # This makes the report reflect only evaluable samples.
                    total -= 1
                    continue

                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]

                # Top-1
                top1_idx = int(torch.argmax(probs).item())
                top1_label = class_names[top1_idx]
                if top1_label == gt_label:
                    top1_correct += 1

                # Top-3
                k = 3 if len(class_names) >= 3 else len(class_names)
                topk_idxs = torch.topk(probs, k).indices.tolist()
                topk_labels = [class_names[int(i)] for i in topk_idxs]
                if gt_label in topk_labels:
                    top3_correct += 1

        elapsed_s = time.perf_counter() - start_eval

        if total <= 0:
            print(f"FAIL: no evaluable images found under {data_root}")
            print("Ensure folders match class names in classes.txt")
            return 4

        top1 = (top1_correct / total) * 100.0
        top3 = (top3_correct / total) * 100.0

        print(f"Imported app/model in: {import_elapsed:.3f}s")
        print(f"Dataset root: {data_root}")
        print(f"Images evaluated: {total}")
        print(f"Top-1 accuracy: {top1_correct}/{total} ({top1:.2f}%)")
        print(f"Top-3 accuracy: {top3_correct}/{total} ({top3:.2f}%)")
        print(f"Eval time (s): {elapsed_s:.3f}")

        return 0

    except Exception as exc:
        print(f"FAIL: evaluation raised {type(exc).__name__}: {exc}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
