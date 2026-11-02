"""Optional first-run downloads. Not used in the recorded cached-data experiment.

Run in a fresh checkout: python prepare_inputs.py --download
This fetches about 1.2 GB of data and weights. The experiment itself stays offline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
FILES = {
    "train": "data/train-00000-of-00001-1028f23e353fbe3e.parquet",
    "validation": "data/validation-00000-of-00001-6c7328ff6c84284c.parquet",
    "test": "data/test-00000-of-00001-f0e719df791966ff.parquet",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true", help="Explicitly enable network downloads")
    args = p.parse_args()
    if not args.download:
        p.error("Use --download to explicitly request downloads, or use existing cache paths in README.")
    from huggingface_hub import hf_hub_download, snapshot_download
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = Path.home() / ".cache/huggingface/datasets/derek-thomas___science_qa/default/0.0.0" / CFG["dataset_cache_revision"]
    data.mkdir(parents=True, exist_ok=True)
    for split, filename in FILES.items():
        dest = data / f"science_qa-{split}.arrow"
        if dest.exists():
            print(f"Keep existing {dest}")
            continue
        source = hf_hub_download(repo_id=CFG["dataset_id"], repo_type="dataset", revision=CFG["dataset_cache_revision"], filename=filename)
        table = pq.read_table(source)
        temp = dest.with_suffix(".arrow.partial")
        with pa.OSFile(str(temp), "wb") as out:
            with pa.ipc.new_stream(out, table.schema) as writer:
                writer.write_table(table)
        temp.replace(dest)
        print(f"Created {dest}: {table.num_rows} rows")
    snapshot_download(repo_id=CFG["text_model_id"], revision=CFG["text_model_revision"],
                      allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "1_Pooling/*", "README.md"])
    weights = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        temp = weights.with_suffix(".partial")
        urllib.request.urlretrieve("https://download.pytorch.org/models/resnet18-f37072fd.pth", temp)
        digest = hashlib.sha256(temp.read_bytes()).hexdigest()
        if not digest.startswith("f37072fd"):
            raise ValueError(f"Unexpected ResNet-18 checksum: {digest}")
        temp.replace(weights)
    print("Inputs prepared. Arrow serialization may differ from the original cache; compare row manifests, not just file hashes.")


if __name__ == "__main__":
    main()
