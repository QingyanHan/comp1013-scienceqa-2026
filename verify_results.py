"""Independent audit of saved predictions and protocol receipts; never trains models."""
import csv
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise AssertionError(label)
    checks.append(label)


checks = []
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--predictions-only', action='store_true', help='Explicitly skip local fitted-model checksum checks')
args = parser.parse_args()
receipt = read("results/test_evaluation_receipt.json")
lock = read("results/selection_lock.json")
check(lock["test_used_for_selection"] is False, "Validation-only model selection declared")
check(lock["locked_utc"] <= receipt["started_utc"], "Selection precedes test evaluation")
check(digest(ROOT / "results/selection_lock.json") == receipt["selection_lock_sha256"], "Selection receipt hash")
for name, expected in receipt["protocol_hashes"].items():
    check(digest(ROOT / name) == expected, f"Frozen source: {name}")
if not args.predictions_only:
    for name, expected in lock["model_hashes"].items():
        check(digest(ROOT / "models" / name) == expected, f"Model checksum: {name}")
with (ROOT / "results/test_metrics.csv").open(newline="", encoding="utf-8") as f:
    expected_metrics = {(r["method"], int(r["seed"])): r for r in csv.DictReader(f)}
with (ROOT / "results/test_manifest.csv").open(newline="", encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))
check(len({r["id"] for r in manifest}) == len(manifest), "Unique test row identifiers")
for name, expected_hash in receipt["prediction_files"].items():
    path = ROOT / "results" / name
    check(digest(path) == expected_hash, f"Prediction checksum: {name}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    method, seed = path.stem.removeprefix("predictions_").rsplit("_", 1)
    expected = expected_metrics[(method, int(seed))]
    check(len(rows) == len(manifest) == receipt["n"], f"Denominator: {name}")
    good, nll, brier = 0, 0.0, 0.0
    conf = [[0] * 4 for _ in range(4)]
    for r, m in zip(rows, manifest):
        y, k, pred = int(r["answer"]), int(r["n_choices"]), int(r["prediction"])
        p = [float(r[f"p{i}"]) for i in range(4)]
        # Explicit exceptions remain active under python -O.
        if not (r["id"] == m["id"] and y == int(m["answer"]) and k == int(m["n_choices"])):
            raise ValueError(f"Label/identifier alignment failed: {name}, {r['id']}")
        if not (0 <= y < k and 0 <= pred < k and all(v == 0 for v in p[k:])):
            raise ValueError(f"Illegal answer option: {name}, {r['id']}")
        if not (all(math.isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p) - 1) < 2e-6):
            raise ValueError(f"Invalid probabilities: {name}, {r['id']}")
        if pred != max(range(4), key=lambda j: p[j]):
            raise ValueError(f"Prediction/argmax mismatch: {name}, {r['id']}")
        good += pred == y
        conf[y][pred] += 1
        # sklearn clips at machine epsilon of the original probability dtype.
        epsilon = 2 ** (-52 if method in ("tfidf", "majority") else -23)
        nll -= math.log(max(epsilon, min(1-epsilon, p[y])))
        brier += sum((v - int(j == y)) ** 2 for j, v in enumerate(p))
    fs = []
    for j in range(4):
        den = sum(conf[j]) + sum(conf[i][j] for i in range(4))
        fs.append(2 * conf[j][j] / den if den else 0.0)
    actual = {"accuracy": good / len(rows), "macro_f1": sum(fs) / 4, "brier": brier / len(rows), "nll": nll / len(rows)}
    for metric, value in actual.items():
        check(abs(value - float(expected[metric])) < 2e-6, f"Recomputed {metric}: {name}")
    checks.append(f"Legal options, finite normalized probabilities and label alignment: {name}")
    check(math.isfinite(float(expected["nll"])), f"Finite NLL: {name}")
for split in ("train", "validation", "test"):
    audit = read(f"results/{split}_data_audit.json")
    check(audit["raw_rows"] == audit["no_image"] + audit["excluded_choice_count"] + audit["included"], f"Exclusions reconcile: {split}")
    check(audit["decode_failures"] == 0, f"Image decoding: {split}")
output = {"checked_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "checks": checks,"model_files_checked":not args.predictions_only,
          "scope": "Independent standard-library recalculation of accuracy, macro F1, Brier, NLL and structural/hash checks."}
(ROOT / ('results/prediction_verification.json' if args.predictions_only else 'results/independent_verification.json')).write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"PASS: {len(checks)} independent checks, {receipt['n']} test rows, {len(receipt['prediction_files'])} prediction files")
