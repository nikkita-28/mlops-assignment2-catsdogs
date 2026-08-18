"""Post-deployment model performance tracking (M5).

Sends a labeled batch of held-out test images to the *deployed* /predict
endpoint, compares predictions with true labels, and writes a small
report with live accuracy and a confusion matrix.

Usage (from repo root, with data/processed/test present):
    python scripts/evaluate_batch.py [--base-url http://localhost:8000] [--per-class 15]
"""

import argparse
import json
import random
from pathlib import Path

# Python puts a script's own directory on sys.path, so this resolves when invoked
# as the documented `python scripts/evaluate_batch.py`.
from smoke_test import post_image, wait_for_health  # reuse the stdlib helpers

CLASS_NAMES = ["cat", "dog"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--out", default="monitoring/post_deploy_report.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    rng = random.Random(args.seed)

    print(f"Waiting for {base_url}/health ...")
    wait_for_health(base_url)

    samples: list[tuple[Path, str]] = []
    for cls in CLASS_NAMES:
        images = sorted((Path(args.test_dir) / cls).glob("*.jpg"))
        if not images:
            raise SystemExit(f"No test images found in {args.test_dir}/{cls}")
        samples += [(p, cls) for p in rng.sample(images, min(args.per_class, len(images)))]

    # confusion[true][predicted]
    confusion = {t: {p: 0 for p in CLASS_NAMES} for t in CLASS_NAMES}
    records = []
    for path, true_label in samples:
        result = post_image(f"{base_url}/predict", path)
        predicted = result["label"]
        confusion[true_label][predicted] += 1
        records.append({"image": path.name, "true": true_label, "predicted": predicted,
                        "probabilities": result["probabilities"]})
        print(f"{path.name}: true={true_label} predicted={predicted}")

    correct = sum(confusion[c][c] for c in CLASS_NAMES)
    accuracy = correct / len(samples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Post-deployment performance report",
        "",
        f"- Endpoint: `{base_url}/predict`",
        f"- Samples: {len(samples)} ({args.per_class} requested per class)",
        f"- **Live accuracy: {accuracy:.2%}** ({correct}/{len(samples)})",
        "",
        "## Confusion matrix (rows = true, columns = predicted)",
        "",
        "| true \\ predicted | " + " | ".join(CLASS_NAMES) + " |",
        "| --- | " + " | ".join("---" for _ in CLASS_NAMES) + " |",
    ]
    for t in CLASS_NAMES:
        lines.append(f"| {t} | " + " | ".join(str(confusion[t][p]) for p in CLASS_NAMES) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({"accuracy": accuracy, "confusion": confusion,
                                     "records": records}, indent=2), encoding="utf-8")

    print(f"\nLive accuracy: {accuracy:.2%} — report written to {out_path} and {json_path}")


if __name__ == "__main__":
    main()
