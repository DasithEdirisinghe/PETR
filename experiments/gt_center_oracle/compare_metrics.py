#!/usr/bin/env python3
"""Compare original nuScenes validation with GT-center-oracle results."""

import argparse
import json

METRICS = (
    ("NDS", True),
    ("mAP", True),
    ("mATE", False),
    ("mASE", False),
    ("mAOE", False),
    ("mAVE", False),
    ("mAAE", False),
)


def extract(data):
    """Return seven metrics from an official or MMDetection3D metric record."""
    if "nd_score" in data and "tp_errors" in data:
        errors = data["tp_errors"]
        return {
            "NDS": float(data["nd_score"]),
            "mAP": float(data["mean_ap"]),
            "mATE": float(errors["trans_err"]),
            "mASE": float(errors["scale_err"]),
            "mAOE": float(errors["orient_err"]),
            "mAVE": float(errors["vel_err"]),
            "mAAE": float(errors["attr_err"]),
        }

    result = {}
    for name, _ in METRICS:
        matches = [
            value for key, value in data.items()
            if key == name or key.endswith("/" + name)
        ]
        if len(matches) != 1:
            return None
        result[name] = float(matches[0])
    return result


def load(path):
    if path.endswith(".log.json"):
        records = []
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    metrics = extract(row)
                    if metrics is not None:
                        records.append((row.get("epoch"), metrics))

        if not records:
            raise ValueError("No complete nuScenes validation record in " + path)

        epoch_24 = [metrics for epoch, metrics in records if epoch == 24]
        if epoch_24:
            return epoch_24[-1]

        print("WARNING: no epoch-24 record in {}; using last metric record"
              .format(path))
        return records[-1][1]

    with open(path) as handle:
        metrics = extract(json.load(handle))
    if metrics is None:
        raise ValueError("Missing nuScenes metrics in " + path)
    return metrics


def table(title, original, oracle):
    print("\n" + title)
    print("{:<7} {:>10} {:>12} {:>10} {:>9}".format(
        "Metric", "Original", "GT-center", "Delta", "Outcome"))
    print("-" * 54)

    for name, higher_is_better in METRICS:
        before, after = original[name], oracle[name]
        delta = after - before
        outcome = (
            "same" if delta == 0 else
            "better" if (delta > 0) == higher_is_better else
            "worse"
        )
        print("{:<7} {:>10.4f} {:>12.4f} {:>+10.4f} {:>9}".format(
            name, before, after, delta, outcome))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--petr-original", required=True)
    parser.add_argument("--petr-oracle", required=True)
    parser.add_argument("--urope-original", required=True)
    parser.add_argument("--urope-oracle", required=True)
    args = parser.parse_args()

    table("PETR baseline",
          load(args.petr_original), load(args.petr_oracle))
    table("PETR + URoPE",
          load(args.urope_original), load(args.urope_oracle))

    print("\nDelta = GT-center minus original. Higher is better for NDS/mAP; "
          "lower is better for error metrics.")
    print("GT-center evaluation uses validation GT during inference.")


if __name__ == "__main__":
    main()
