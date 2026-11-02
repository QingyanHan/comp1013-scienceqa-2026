"""Create figures and descriptive summaries from locked, saved results only."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)
read = lambda name: json.loads((R / name).read_text(encoding="utf-8"))
scores = pd.read_csv(R / "test_metrics.csv")
lock = read("selection_lock.json")["selections"]
seed = 42
order = ["majority", "tfidf", "image", "text", "early", "late"]
names = {"majority":"Majority", "tfidf":"TF-IDF + LR", "image":"Image only", "text":"Text only", "early":"Early fusion", "late":"Late fusion"}
colours = {"majority":"#9ca3af", "tfidf":"#64748b", "image":"#94a3b8", "text":"#2171b5", "early":"#117866", "late":"#e28c2f"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})


def save(fig, stem):
    fig.savefig(F / f"{stem}.png", bbox_inches="tight")
    fig.savefig(F / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


primary = scores[scores.seed == seed].set_index("method").loc[order]
fig, ax = plt.subplots(figsize=(8, 3.2))
bars = ax.bar(range(6), primary.accuracy * 100, color=[colours[k] for k in order], width=.65)
ax.set_xticks(range(6), [names[k] for k in order])
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, max(65, primary.accuracy.max()*100+10))
random_acc = read("test_data_audit.json")["uniform_guess_expected_accuracy"]
ax.axhline(random_acc*100, color="#6b7280", linestyle=":", label=f"Uniform guess expectation {random_acc*100:.1f}%")
ax.bar_label(bars, fmt="%.2f", padding=4)
ax.legend(loc="upper left", frameon=False, fontsize=9)
save(fig, "accuracy_comparison")

fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.8), sharey=True)
gaps = []
for ax, kind in zip(axes, ["text", "image", "early"]):
    curves = pd.read_csv(R / f"{kind}_42_curves.csv")
    selected = lock[f"{kind}_42"]
    ax.plot(curves.epoch, curves.train_nll, label="Train", color="#2171b5")
    ax.plot(curves.epoch, curves.validation_nll, label="Validation", color="#d67519")
    ax.axvline(selected["best_epoch"], color="#6b7280", linestyle=":", label="Selected epoch")
    ax.set_title(names[kind]); ax.set_xlabel("Epoch")
    gaps.append({"method":kind,"epoch":selected["best_epoch"],"train_accuracy":selected["train"]["accuracy"],"validation_accuracy":selected["validation"]["accuracy"],"gap":selected["train"]["accuracy"]-selected["validation"]["accuracy"],"train_nll":selected["train"]["nll"],"validation_nll":selected["validation"]["nll"]})
axes[0].set_ylabel("Negative log-likelihood")
axes[-1].legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig,"learning_curves")

predictions = {k:pd.read_csv(R / f"predictions_{k}_42.csv") for k in order}
fig, axes = plt.subplots(1,2,figsize=(7.2,3))
for ax, kind in zip(axes,["text","early"]):
    frame=predictions[kind]
    conf=np.zeros((4,4),dtype=int)
    np.add.at(conf,(frame.answer,frame.prediction),1)
    ax.imshow(conf,cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(conf[i,j]),ha="center",va="center",color="white" if conf[i,j]>conf.max()*.55 else "#102435")
    ax.set_xticks(range(4),list("ABCD")); ax.set_yticks(range(4),list("ABCD"))
    ax.set_xlabel("Predicted position"); ax.set_ylabel("True position"); ax.set_title(names[kind])
fig.tight_layout();save(fig,"confusion_matrices")

groups=pd.read_csv(R/"group_metrics.csv")
subjects=groups[(groups.field=="subject") & (groups.method.isin(["text","early","late"]))]
fig,ax=plt.subplots(figsize=(7.6,2.8))
subnames=list(subjects.group.unique())
for i,kind in enumerate(["text","early","late"]):
    df=subjects[subjects.method==kind].set_index("group").loc[subnames]
    ax.bar(np.arange(len(subnames))+(i-1)*.23,df.accuracy*100,.23,label=names[kind],color=colours[kind])
ax.set_xticks(range(len(subnames)),[f"{s.replace(' science',' science')}\n(n={int(subjects[subjects.group==s].iloc[0]['n'])})" for s in subnames])
ax.set_ylabel("Test accuracy (%)");ax.legend(frameon=False,loc="upper right",ncol=3,fontsize=8);ax.set_ylim(0,100)
save(fig,"subject_accuracy")

meta=json.loads((ROOT/"cache/test_metadata.json").read_text(encoding="utf-8"))
early=predictions["early"];text=predictions["text"]
errors=list(np.flatnonzero(early.prediction.to_numpy()!=early.answer.to_numpy())[:2])
corrected=np.flatnonzero((early.prediction.to_numpy()==early.answer.to_numpy()) & (text.prediction.to_numpy()!=text.answer.to_numpy()))
chosen=[("early_error",int(i)) for i in errors]
if len(corrected):chosen.append(("fusion_corrected",int(corrected[0])))
cfg=json.loads((ROOT/"config.json").read_text())
import os
data=Path(os.environ.get("SCIENCEQA_DATA_DIR",str(Path.home()/".cache/huggingface/datasets/derek-thomas___science_qa/default/0.0.0"/cfg["dataset_cache_revision"])))
with pa.memory_map(str(data/"science_qa-test.arrow"),"r") as f:
    table=pa.ipc.open_stream(f).read_all().select(["image"])
examples=[]
for condition, i in chosen:
    m=meta[i]
    raw=table.column("image")[m["source_index"]].as_py()
    image_data=raw["bytes"] if raw["bytes"] is not None else Path(raw["path"]).read_bytes()
    with Image.open(io.BytesIO(image_data)) as im:
        im.convert("RGB").save(F/f"{m['id']}.png")
    examples.append({**m,"selection":condition,"early_prediction":int(early.iloc[i].prediction),"text_prediction":int(text.iloc[i].prediction),"image_file":f"figures/{m['id']}.png","early_probabilities":[float(early.iloc[i][f'p{j}']) for j in range(4)]})

seeds=scores[scores.method.isin(["text","image","early","late"])].groupby("method").agg(accuracy_mean=("accuracy","mean"),accuracy_sd=("accuracy","std"),accuracy_min=("accuracy","min"),accuracy_max=("accuracy","max"),n_seeds=("seed","size"))
seeds.to_csv(R/"seed_summary.csv")
pd.DataFrame(gaps).to_csv(R/"overfitting_summary.csv",index=False)
summary={"primary":primary.reset_index().to_dict(orient="records"),"seed_summary":seeds.reset_index().to_dict(orient="records"),"overfitting":gaps,"examples":examples,"figure_note":"Plots use saved predictions and selection records. No new fitting or test selection."}
(R/"analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"examples":[{k:e[k] for k in ['id','question','choices','answer','early_prediction','text_prediction']} for e in examples],"seed_summary":summary["seed_summary"]},ensure_ascii=False,indent=2))
