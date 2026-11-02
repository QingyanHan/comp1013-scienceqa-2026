"""CPU-only, staged ScienceQA experiment. See PROTOCOL.md before running.

Commands: prepare, features, train, evaluate. Each stage saves its evidence.
No stage downloads data/models, sends messages or submits coursework.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
for directory in ("cache", "models", "results", "figures", "private", "logs", "delivery"):
    (ROOT / directory).mkdir(exist_ok=True)
torch.set_num_threads(CFG["threads"])
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
threadpool_limits(CFG["threads"])


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def current_protocol():
    return {name: sha(ROOT / name) for name in ("config.json", "PROTOCOL.md", "run_project.py")}


def check_protocol():
    expected = read_json(ROOT / "results/protocol_lock.json")["hashes"]
    if expected != current_protocol():
        raise RuntimeError("Protocol/source changed since prepare; record a repair and use a new run directory.")


def data_root():
    return Path(os.environ.get("SCIENCEQA_DATA_DIR", str(
        Path.home() / ".cache/huggingface/datasets/derek-thomas___science_qa/default/0.0.0"
        / CFG["dataset_cache_revision"])))


def text_model_root():
    return Path(os.environ.get("SCIENCEQA_TEXT_MODEL_DIR", str(
        Path.home() / ".cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots"
        / CFG["text_model_revision"])))


def source_rows(split):
    """Read explicit cached Arrow files. Never silently fetch or replace data."""
    path = data_root() / f"science_qa-{split}.arrow"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Follow README data preparation instructions.")
    with pa.memory_map(str(path), "r") as f:
        table = pa.ipc.open_stream(f).read_all()
    # Read only declared inputs, target and diagnostic metadata; ignore rationales.
    names = ["image", "question", "hint", "choices", "answer", "subject", "grade", "topic"]
    return table.select(names).to_pylist()


def normalise(text):
    return re.sub(r"\s+", " ", text or "").strip()


def joined_text(row):
    choices = " ".join(f"{chr(65+i)}: {s}" for i, s in enumerate(row["choices"]))
    return f"Question: {row['question']} Choices: {choices} Hint: {row['hint']}"


def image_bytes(row):
    img = row["image"]
    if img.get("bytes") is not None:
        return img["bytes"]
    if img.get("path"):
        return Path(img["path"]).read_bytes()
    raise ValueError("Image structure present without bytes or readable path")


def prepare_split(split):
    path = ROOT / f"cache/{split}_metadata.json"
    if path.exists():
        return read_json(path)
    source = source_rows(split)
    rows, stats = [], {"raw_rows": len(source), "no_image": 0, "excluded_choice_count": 0}
    t0 = time.perf_counter()
    for index, row in enumerate(source):
        if row["image"] is None:
            stats["no_image"] += 1
            continue
        if not 2 <= len(row["choices"]) <= 4:
            stats["excluded_choice_count"] += 1
            continue
        if not 0 <= row["answer"] < len(row["choices"]):
            raise ValueError(f"Invalid label at {split}_{index}")
        raw = image_bytes(row)
        with Image.open(io.BytesIO(raw)) as im:
            im.load()  # Included corrupt images stop the audit.
            width, height, mode = *im.size, im.mode
        clean = {k: row[k] for k in ("answer", "subject", "grade", "topic")}
        clean.update({"id": f"{split}_{index:05d}", "source_index": index,
                      "question": normalise(row["question"]), "hint": normalise(row["hint"]),
                      "choices": [normalise(c) for c in row["choices"]],
                      "n_choices": len(row["choices"]), "width": width, "height": height,
                      "mode": mode, "image_sha256": hashlib.sha256(raw).hexdigest()})
        clean["text_sha256"] = hashlib.sha256(joined_text(clean).encode()).hexdigest()
        rows.append(clean)
    stats.update({"included": len(rows), "decode_failures": 0,
                  "choice_counts": dict(Counter(r["n_choices"] for r in rows)),
                  "answer_counts": dict(Counter(r["answer"] for r in rows)),
                  "subject_counts": dict(Counter(r["subject"] for r in rows)),
                  "grade_counts": dict(Counter(r["grade"] for r in rows)),
                  "empty_hint": sum(not r["hint"] for r in rows),
                  "unique_image_hashes": len({r["image_sha256"] for r in rows}),
                  "unique_text_hashes": len({r["text_sha256"] for r in rows}),
                  "uniform_guess_expected_accuracy": float(np.mean([1/r["n_choices"] for r in rows])),
                  "audit_seconds": time.perf_counter()-t0,
                  "source_arrow_sha256": sha(data_root()/f"science_qa-{split}.arrow")})
    save_json(path, rows)
    save_json(ROOT / f"results/{split}_data_audit.json", stats)
    pd.DataFrame([{k: r[k] for k in ("id", "source_index", "answer", "n_choices", "subject", "grade", "image_sha256", "text_sha256")} for r in rows]).to_csv(ROOT/f"results/{split}_manifest.csv", index=False)
    print(f"Prepared {split}: {stats}", flush=True)
    return rows


def prepare():
    if (ROOT/"results/protocol_lock.json").exists():
        check_protocol()
    else:
        save_json(ROOT/"results/protocol_lock.json", {"created_utc": stamp(), "hashes": current_protocol(), "test_evaluated": False})
    versions = {}
    for name in ("numpy", "pandas", "torch", "torchvision", "transformers", "sentence-transformers", "scikit-learn", "pyarrow", "pillow", "joblib", "matplotlib"):
        versions[name] = importlib.metadata.version(name)
    save_json(ROOT/"results/environment.json", {"created_utc": stamp(), "python": sys.version,
              "platform": platform.platform(), "processor": platform.processor(),
              "device": "cpu", "threads": CFG["threads"], "versions": versions})
    (ROOT/"requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in versions.items())+"\n", encoding="utf-8")
    tr, va = prepare_split("train"), prepare_split("validation")
    save_json(ROOT/"results/train_validation_overlap.json", {
        field: {"validation_rows_seen_in_train": sum(r[field] in {t[field] for t in tr} for r in va)}
        for field in ("image_sha256", "text_sha256")})


def encode_split(split):
    check_protocol()
    metadata = prepare_split(split)
    image_out, text_out = ROOT/f"cache/{split}_image.npy", ROOT/f"cache/{split}_text.npy"
    timing_file = ROOT/f"results/{split}_feature_timing.json"
    timing = read_json(timing_file) if timing_file.exists() else {"device": "cpu", "threads": CFG["threads"]}
    if not image_out.exists():
        from torchvision.models import resnet18
        from torchvision.transforms import functional as F
        weights = Path(os.environ.get("SCIENCEQA_RESNET_WEIGHTS", str(Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")))
        if not weights.exists():
            raise FileNotFoundError(weights)
        net = resnet18(weights=None)
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        net.fc = nn.Identity()
        net.eval()
        source = source_rows(split)
        arrays = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(metadata), CFG["image_batch"]):
                batch = []
                for m in metadata[start:start+CFG["image_batch"]]:
                    with Image.open(io.BytesIO(image_bytes(source[m["source_index"]]))) as im:
                        im = ImageOps.pad(im.convert("RGB"), (CFG["image_size"],)*2, method=Image.Resampling.BILINEAR, color="white")
                        batch.append(F.normalize(F.to_tensor(im), [0.485,0.456,0.406], [0.229,0.224,0.225]))
                arrays.append(net(torch.stack(batch)).numpy())
                if start % 512 == 0:
                    print(f"{split} image {start}/{len(metadata)} {time.perf_counter()-t0:.1f}s", flush=True)
        image_features = np.concatenate(arrays).astype("float32")
        np.save(image_out, image_features)
        timing.update({"image_seconds": time.perf_counter()-t0, "image_rows": len(metadata),
                       "image_weight_sha256": sha(weights), "image_weight_bytes": weights.stat().st_size,
                       "image_feature_sha256": sha(image_out)})
        save_json(timing_file,timing)
        del source, net, arrays
    if not text_out.exists():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(text_model_root()), device="cpu", local_files_only=True)
        model.max_seq_length = CFG["text_max_tokens"]
        sequences = [[f"Question: {m['question']} Hint: {m['hint']}", *m["choices"]] for m in metadata]
        unique = list(dict.fromkeys(s for row in sequences for s in row))
        mapping = {s:i for i,s in enumerate(unique)}
        lengths = [len(x) for x in model.tokenizer(unique, truncation=False, padding=False)["input_ids"]]
        t0 = time.perf_counter()
        embeddings = []
        for start in range(0,len(unique),256):
            embeddings.append(model.encode(unique[start:start+256], batch_size=CFG["text_batch"], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True))
            print(f"{split} text {min(start+256,len(unique))}/{len(unique)} {time.perf_counter()-t0:.1f}s", flush=True)
        embeddings = np.concatenate(embeddings)
        X = np.zeros((len(metadata),5,384),dtype="float32")
        for i,row in enumerate(sequences):
            X[i,:len(row)] = embeddings[[mapping[s] for s in row]]
        np.save(text_out,X.reshape(len(metadata),-1))
        timing.update({"text_seconds": time.perf_counter()-t0, "unique_text_sequences": len(unique),
                       "text_rows": len(metadata), "unique_sequences_truncated": sum(n>CFG["text_max_tokens"] for n in lengths),
                       "question_hint_rows_truncated": sum(lengths[mapping[row[0]]]>CFG["text_max_tokens"] for row in sequences),
                       "choice_sequences_truncated": sum(lengths[mapping[s]]>CFG["text_max_tokens"] for row in sequences for s in row[1:]),
                       "text_weight_sha256": sha(text_model_root()/"model.safetensors"),
                       "text_weight_bytes": (text_model_root()/"model.safetensors").stat().st_size,
                       "text_feature_sha256": sha(text_out)})
        save_json(timing_file,timing)
    Xim, Xtxt = np.load(image_out), np.load(text_out)
    assert Xim.shape == (len(metadata),512) and Xtxt.shape == (len(metadata),1920)
    assert np.isfinite(Xim).all() and np.isfinite(Xtxt).all()
    return metadata, Xim, Xtxt


def features():
    for split in ("train","validation"):
        encode_split(split)


def masked_probs(p, counts):
    p = np.asarray(p,dtype="float64").copy()
    p[np.arange(4)[None,:] >= np.asarray(counts)[:,None]] = 0
    den = p.sum(axis=1,keepdims=True)
    if np.any(den<=0):
        raise ValueError("No mass on legal choices")
    return p/den


def metrics(y,p):
    pred = p.argmax(1)
    return {"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",labels=list(range(4)),zero_division=0)),
            "nll":float(log_loss(y,p,labels=list(range(4)))),
            "brier":float(np.mean(np.sum((p-np.eye(4)[y])**2,axis=1)))}


class Head(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,CFG["hidden_dim"]), nn.ReLU(), nn.Dropout(CFG["dropout"]), nn.Linear(CFG["hidden_dim"],4))

    def forward(self,x,counts):
        return self.net(x).masked_fill(torch.arange(4)[None,:]>=counts[:,None],-1e9)


def probs_head(model,X,counts):
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(torch.as_tensor(X,dtype=torch.float32),torch.as_tensor(counts,dtype=torch.long)),1).numpy()


def train_head(kind,seed,Xtr,Xva,tr,va):
    target = ROOT/f"models/{kind}_{seed}.pt"
    if target.exists():
        return read_json(ROOT/f"results/{kind}_{seed}_selection.json")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Head(Xtr.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(),lr=CFG["learning_rate"],weight_decay=CFG["weight_decay"])
    xt = torch.from_numpy(Xtr); yt = torch.tensor([m["answer"] for m in tr]); kt = torch.tensor([m["n_choices"] for m in tr])
    yv = np.array([m["answer"] for m in va]); kv = np.array([m["n_choices"] for m in va])
    best,stale,history,best_state = float("inf"),0,[],None
    t0 = time.perf_counter()
    for epoch in range(1,CFG["max_epochs"]+1):
        model.train()
        order=torch.randperm(len(xt))
        for idx in order.split(CFG["train_batch"]):
            optimiser.zero_grad()
            loss=nn.functional.cross_entropy(model(xt[idx],kt[idx]),yt[idx])
            loss.backward(); optimiser.step()
        mt=metrics(yt.numpy(),probs_head(model,Xtr,kt.numpy()))
        mv=metrics(yv,probs_head(model,Xva,kv))
        history.append({"epoch":epoch,**{f"train_{k}":v for k,v in mt.items()},**{f"validation_{k}":v for k,v in mv.items()}})
        if mv["nll"] < best-CFG["min_delta"]:
            best,stale,best_epoch,best_state = mv["nll"],0,epoch,copy.deepcopy(model.state_dict())
        else:
            stale+=1
        print(f"{kind} seed{seed} epoch{epoch} train_acc={mt['accuracy']:.4f} val_acc={mv['accuracy']:.4f} val_nll={mv['nll']:.4f}",flush=True)
        if stale>=CFG["patience"]:
            break
    elapsed=time.perf_counter()-t0
    torch.save({"state_dict":best_state,"input_dim":Xtr.shape[1],"seed":seed,"best_epoch":best_epoch},target)
    model.load_state_dict(best_state)
    pt,pv=probs_head(model,Xtr,kt.numpy()),probs_head(model,Xva,kv)
    np.save(ROOT/f"cache/{kind}_{seed}_validation_probs.npy",pv)
    pd.DataFrame(history).to_csv(ROOT/f"results/{kind}_{seed}_curves.csv",index=False)
    selected={"kind":kind,"seed":seed,"best_epoch":best_epoch,"epochs_run":epoch,"training_seconds":elapsed,
              "head_parameters":sum(p.numel() for p in model.parameters()),"checkpoint_bytes":target.stat().st_size,
              "train":metrics(yt.numpy(),pt),"validation":metrics(yv,pv)}
    save_json(ROOT/f"results/{kind}_{seed}_selection.json",selected)
    return selected


def matrices(split,scalers):
    im,tx=np.load(ROOT/f"cache/{split}_image.npy"),np.load(ROOT/f"cache/{split}_text.npy")
    im=scalers["image"].transform(im).astype("float32")
    tx=scalers["text"].transform(tx).astype("float32")
    return {"image":im,"text":tx,"early":np.concatenate([im,tx],axis=1)}


def train():
    check_protocol()
    if (ROOT/"results/test_evaluation_receipt.json").exists():
        raise RuntimeError("Test has been evaluated: training is locked")
    if (ROOT/"results/selection_lock.json").exists():
        print("Selection already locked; no training repeated",flush=True); return
    tr=read_json(ROOT/"cache/train_metadata.json"); va=read_json(ROOT/"cache/validation_metadata.json")
    scalers={k:StandardScaler().fit(np.load(ROOT/f"cache/train_{k}.npy")) for k in ("image","text")}
    joblib.dump(scalers,ROOT/"models/scalers.joblib")
    Xtr,Xva=matrices("train",scalers),matrices("validation",scalers)
    ytr=np.array([r["answer"] for r in tr]); yva=np.array([r["answer"] for r in va]); kva=np.array([r["n_choices"] for r in va])
    selections={}
    for seed in CFG["seeds"]:
        for kind in ("text","image","early"):
            selections[f"{kind}_{seed}"]=train_head(kind,seed,Xtr[kind],Xva[kind],tr,va)
        pt=np.load(ROOT/f"cache/text_{seed}_validation_probs.npy"); pi=np.load(ROOT/f"cache/image_{seed}_validation_probs.npy")
        rows=[{"text_weight":w,**metrics(yva,w*pt+(1-w)*pi)} for w in CFG["late_text_weight_grid"]]
        selected=min(rows,key=lambda r:(r["nll"],r["text_weight"]))
        selections[f"late_{seed}"]=selected
        save_json(ROOT/f"results/late_{seed}_weight_search.json",rows)
    t0=time.perf_counter()
    vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    Ztr=vec.fit_transform([joined_text(r) for r in tr]); Zva=vec.transform([joined_text(r) for r in va])
    trials=[]; best=None
    for c in CFG["tfidf_C_grid"]:
        clf=LogisticRegression(max_iter=2000,C=c,random_state=42)
        start=time.perf_counter(); clf.fit(Ztr,ytr)
        p=masked_probs(clf.predict_proba(Zva),kva)
        trial={"C":c,"fit_seconds":time.perf_counter()-start,**metrics(yva,p)}; trials.append(trial)
        if best is None or trial["nll"] < best[0]:
            best=(trial["nll"],clf,trial)
    joblib.dump({"vectorizer":vec,"classifier":best[1]},ROOT/"models/tfidf.joblib")
    selections["tfidf"]={**best[2],"search_total_seconds":time.perf_counter()-t0,"train":metrics(ytr,masked_probs(best[1].predict_proba(Ztr),[r["n_choices"] for r in tr]))}
    save_json(ROOT/"results/tfidf_validation_search.json",trials)
    counts=np.bincount(ytr,minlength=4)
    selections["majority_index"]=int(counts.argmax())
    lock={"locked_utc":stamp(),"protocol_hashes":current_protocol(),"test_used_for_selection":False,
          "selections":selections,"model_hashes":{p.name:sha(p) for p in sorted((ROOT/"models").glob('*'))}}
    save_json(ROOT/"results/selection_lock.json",lock)
    print("MODEL SELECTION LOCKED",flush=True)


def load_head(kind,seed):
    saved=torch.load(ROOT/f"models/{kind}_{seed}.pt",map_location="cpu",weights_only=True)
    model=Head(saved["input_dim"]); model.load_state_dict(saved["state_dict"]); model.eval()
    return model


def evaluate():
    check_protocol()
    receipt=ROOT/"results/test_evaluation_receipt.json"
    if receipt.exists():
        print("Test already evaluated. Reuse saved predictions; no repeat.",flush=True); return
    lock=read_json(ROOT/"results/selection_lock.json")
    assert lock["protocol_hashes"]==current_protocol()
    assert lock["model_hashes"]=={p.name:sha(p) for p in sorted((ROOT/"models").glob('*'))}
    started=stamp()
    meta,_,_=encode_split("test")
    tr=read_json(ROOT/"cache/train_metadata.json")
    scalers=joblib.load(ROOT/"models/scalers.joblib"); X=matrices("test",scalers)
    y=np.array([m["answer"] for m in meta]); k=np.array([m["n_choices"] for m in meta])
    scores,perf,probabilities=[],[],{}
    for seed in CFG["seeds"]:
        for kind in ("text","image","early"):
            model=load_head(kind,seed); p=probs_head(model,X[kind],k)
            probabilities[(kind,seed)]=p
            repetitions=[]
            for _ in range(20):
                t0=time.perf_counter(); probs_head(model,X[kind],k); repetitions.append(time.perf_counter()-t0)
            perf.append({"method":kind,"seed":seed,"cached_inference_ms_per_sample":1000*float(np.median(repetitions))/len(y),"repeat_count":20})
        w=lock["selections"][f"late_{seed}"]["text_weight"]
        probabilities[("late",seed)]=w*probabilities[("text",seed)]+(1-w)*probabilities[("image",seed)]
    baseline=joblib.load(ROOT/"models/tfidf.joblib")
    t0=time.perf_counter(); Z=baseline["vectorizer"].transform([joined_text(r) for r in meta]); p=masked_probs(baseline["classifier"].predict_proba(Z),k)
    perf.append({"method":"tfidf","seed":42,"text_vectorise_and_inference_ms_per_sample":1000*(time.perf_counter()-t0)/len(y)})
    probabilities[("tfidf",42)]=p
    majority=np.zeros((len(y),4)); majority[:,lock["selections"]["majority_index"]]=1
    probabilities[("majority",42)]=masked_probs(majority,k)
    for (kind,seed),p in probabilities.items():
        assert np.all(p.argmax(1)<k) and np.allclose(p.sum(1),1)
        scores.append({"method":kind,"seed":seed,"n":len(y),**metrics(y,p)})
        frame=pd.DataFrame({"id":[m["id"] for m in meta],"answer":y,"n_choices":k,"prediction":p.argmax(1),**{f"p{j}":p[:,j] for j in range(4)}})
        frame.to_csv(ROOT/f"results/predictions_{kind}_{seed}.csv",index=False,float_format="%.10g")
    # Predeclared test diagnostic: shuffle visual features within choice-count strata.
    rng=np.random.default_rng(2026); perm=np.arange(len(y))
    for count in np.unique(k):
        idx=np.flatnonzero(k==count); perm[idx]=rng.permutation(idx)
    seed=CFG["primary_seed"]
    shuffled_image=probs_head(load_head("image",seed),X["image"][perm],k)
    early_p=probs_head(load_head("early",seed),np.concatenate([X["image"][perm],X["text"]],axis=1),k)
    w=lock["selections"][f"late_{seed}"]["text_weight"]
    late_p=w*probabilities[("text",seed)]+(1-w)*shuffled_image
    ablation=[{"method":kind,"condition":"image_permuted_within_choice_count",**metrics(y,p)} for kind,p in [("early",early_p),("late",late_p)]]
    pd.DataFrame(ablation).to_csv(ROOT/"results/image_shuffle_ablation.csv",index=False)
    overlap={}
    for field in ("image_sha256","text_sha256"):
        train_set={m[field] for m in tr}; seen=np.array([m[field] in train_set for m in meta])
        overlap[field]={"seen_rows":int(seen.sum()),"unseen_rows":int((~seen).sum()),"unseen_metrics":{kind:metrics(y[~seen],probabilities[(kind,seed)][~seen]) for kind in ("text","image","early","late") if (~seen).any()}}
    save_json(ROOT/"results/test_overlap_diagnostics.json",overlap)
    intervals={}
    for kind in ("early","late"):
        diff=(probabilities[(kind,seed)].argmax(1)==y).astype(float)-(probabilities[("text",seed)].argmax(1)==y)
        rng=np.random.default_rng(2026)
        draws=np.array([diff[rng.integers(0,len(y),len(y))].mean() for _ in range(CFG["bootstrap_repetitions"])])
        intervals[kind]={"accuracy_difference":float(diff.mean()),"ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"repetitions":CFG["bootstrap_repetitions"],"unit":"question","caveat":"Descriptive IID bootstrap; repeated images/templates can violate independence."}
    save_json(ROOT/"results/paired_bootstrap.json",intervals)
    groups=[]
    for field in ("subject","grade","n_choices"):
        vals=np.array([str(m[field]) for m in meta])
        for value in sorted(set(vals)):
            mask=vals==value
            for kind in ("text","image","early","late"):
                groups.append({"field":field,"group":value,"method":kind,"n":int(mask.sum()),**metrics(y[mask],probabilities[(kind,seed)][mask])})
    pd.DataFrame(groups).to_csv(ROOT/"results/group_metrics.csv",index=False)
    pd.DataFrame(scores).to_csv(ROOT/"results/test_metrics.csv",index=False)
    pd.DataFrame(perf).to_csv(ROOT/"results/inference_timing.csv",index=False)
    save_json(receipt,{"started_utc":started,"completed_utc":stamp(),"selection_lock_sha256":sha(ROOT/"results/selection_lock.json"),"protocol_hashes":current_protocol(),"n":len(y),"test_used_for_selection":False,"prediction_files":{p.name:sha(p) for p in sorted((ROOT/"results").glob('predictions_*.csv'))}})
    print(pd.DataFrame(scores).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","features","train","evaluate"])
    args=parser.parse_args()
    t0=time.perf_counter(); start=stamp()
    try:
        globals()[args.stage]()
        status="success"
    except Exception:
        status="failed"
        traceback.print_exc()
        raise
    finally:
        save_json(ROOT/f"logs/{args.stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json",
                  {"stage":args.stage,"started_utc":start,"completed_utc":stamp(),"status":status,"seconds":time.perf_counter()-t0})


if __name__=="__main__":
    main()
