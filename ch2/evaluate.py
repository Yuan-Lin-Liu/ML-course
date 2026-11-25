"""Standalone testing: uv run python evaluate.py --fraction 0.8."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from datetime import datetime
import random
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights

import secrets
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay)

def load_images(root, size):
    """Return serializable rows and a separate resized-image cache in one pass."""
    rows, images = [], {}
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        name = path.name.lower()
        if name.startswith("normal"):
            label = 0
        elif name.startswith("disease"):
            label = 1
        else:
            raise ValueError(f"Unknown label: {path}")
        key = str(path.resolve())
        with Image.open(path) as source:
            raw = source.convert("RGB")
            digest = hashlib.sha256(np.asarray(raw).tobytes() + str(raw.size).encode()).hexdigest()
            images[key] = ImageOps.exif_transpose(source).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        rows.append({"path": key, "label": label, "sha256_pixels": digest})
    if not rows:
        raise ValueError(f"No images in {root}")
    return rows, images

class FundusDataset(Dataset):
    def __init__(self, rows, size, augment=False, images=None):
        self.rows = rows
        self.images = []
        for row in rows:
            if images is not None:
                self.images.append(images[row["path"]])
            else:
                with Image.open(row["path"]) as source:
                    self.images.append(ImageOps.exif_transpose(source).convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
        ops = [T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(.1, .1, .1, .02)] if augment else []
        self.transform = T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.transform(self.images[index]), torch.tensor(self.rows[index]["label"], dtype=torch.float32)

def build_model(pretrained=True):
    model=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc=torch.nn.Linear(model.fc.in_features,1)
    return model

def create_run_directory(base):
    """Treat configured output as a prefix; never overwrite an existing run."""
    base=Path(base)
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    for suffix in range(10000):
        name=f'{base.name}_{stamp}'+(f'_{suffix:03d}' if suffix else '')
        candidate=base.parent/name
        try:
            candidate.mkdir(parents=True,exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError('Could not allocate unique run directory')

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state

def predict_batches(model, loader, device):
    """All inference uses FP32 and unweighted mean BCE, including single images."""
    model.eval()
    labels_all, probabilities = [], []
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images).flatten().float()
        loss_sum += criterion(logits, labels).item()
        labels_all.extend(labels.cpu().tolist())
        probabilities.extend(logits.sigmoid().cpu().tolist())
    if not labels_all:
        raise ValueError("No images to predict")
    return loss_sum / len(labels_all), np.asarray(labels_all), np.asarray(probabilities)

def resolve_test_directory(root, checkpoint_path, override=None):
    root = Path(root).resolve()
    if override is not None:
        return root / override
    requested = root / checkpoint_path
    legacy = root / '2-MedImage-TestSet'
    relocated = root / 'data' / '2-MedImage-TestSet'
    if requested.resolve() == legacy and not requested.exists() and relocated.is_dir():
        return relocated
    return requested

def latest_completed_run(root):
    """Ignore incomplete/failed runs when selecting a model for prediction."""
    candidates=[p.parent for p in Path(root).glob('*/metrics.json') if (p.parent/'best.pt').is_file()]
    if not candidates:
        raise FileNotFoundError('No completed run found; train a model or provide --checkpoint')
    return max(candidates,key=lambda p:(p/'metrics.json').stat().st_mtime_ns)

def classification_metrics(labels,probabilities):
    y=np.asarray(labels); p=np.asarray(probabilities); pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        precision=float(precision_score(y,pred,zero_division=0)),
        sensitivity=float(recall_score(y,pred,zero_division=0)),
        specificity=float(tn/(tn+fp)),f1=float(f1_score(y,pred,zero_division=0)),
        roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=.5)

def save_evaluation(y,p,metrics,out):
    y,p=np.asarray(y),np.asarray(p)
    fpr,tpr,_=roc_curve(y,p); fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr,label=f"AUC = {metrics['roc_auc']:.4f}"); ax.plot([0,1],[0,1],'k--')
    ax.set(xlabel='False positive rate',ylabel='Sensitivity',title='Held-out test ROC'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'roc.png',dpi=160); plt.close(fig)
    precision,recall,_=precision_recall_curve(y,p); fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(recall,precision,label=f"AP = {metrics['pr_auc']:.4f}")
    ax.set(xlabel='Recall',ylabel='Precision',title='Held-out test precision-recall'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'precision_recall.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(y,(p>=.5).astype(int),labels=[0,1],display_labels=['Normal','Disease'],ax=ax,colorbar=False)
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)

def write_report(out, metrics, details=None, title="模型评估结果"):
    out = Path(out)
    details = details or {}
    lines = [title] + [f"{key}: {value}" for key, value in details.items()]
    lines += [f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items()]
    lines.append("患病=1，阈值0.5；loss口径见loss_definition或实验说明。")
    lines.append("重复抽取同一测试集不构成新的外部验证，不应反复抽样挑最高分。")
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    page = '<!doctype html><meta charset="utf-8"><title>' + html.escape(title) + '</title><style>body{max-width:1000px;margin:30px auto;font:16px/1.8 sans-serif}pre{white-space:pre-wrap}img{max-width:100%}</style>'
    page += '<h1>' + html.escape(title) + '</h1><pre>' + html.escape(summary) + '</pre>'
    for name in ["loss", "accuracy", "roc", "precision_recall", "confusion_matrix"]:
        if (out / f"{name}.png").exists():
            page += f'<figure><img src="{name}.png"><figcaption>{name}</figcaption></figure>'
    readme = ROOT / "README.md"
    try:
        link = Path(os.path.relpath(readme, out)).as_posix()
    except ValueError:  # A custom output may be on another Windows drive.
        link = readme.as_uri()
    page += '<p><a href="' + html.escape(link, quote=True) + '">操作指令</a></p>'
    (out / "results.html").write_text(page, encoding="utf-8")
    return summary

def save_results(out, metrics, rows, probabilities, details=None, filename="predictions.csv", title="模型评估结果"):
    labels = [row["label"] for row in rows]
    save_evaluation(labels, probabilities, metrics, out)
    with (out / filename).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "label", "disease_probability", "prediction", "correct"])
        for row, probability in zip(rows, probabilities):
            predicted = int(probability >= .5)
            writer.writerow([row["path"], row["label"], float(probability), predicted, predicted == row["label"]])
    summary = write_report(out, metrics, details, title)
    # Completion marker written after all result files have been saved.
    write_json(out / "metrics.json", metrics)
    return summary

def select_subset(rows, fraction, seed):
    """Uniform sampling without replacement; no selection by class or result."""
    if not 0 < fraction <= 1:
        raise ValueError('--fraction must be in (0, 1]')
    count = int(len(rows) * fraction)
    if count < 2:
        raise ValueError('At least two images are needed for evaluation')
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    selected = [rows[i] for i in indices]
    if len({row['label'] for row in selected}) < 2:
        raise ValueError('Sample contains only one class; AUC is undefined. Increase --fraction.')
    return selected

def main():
    parser=argparse.ArgumentParser(description="Evaluate a test subset or predict a single image.")
    parser.add_argument("--checkpoint",help="Default: latest completed training model")
    parser.add_argument("--fraction",type=float,default=.8)
    parser.add_argument("--seed",type=int,help="Omit for a fresh random sample")
    parser.add_argument("--test-dir")
    parser.add_argument("--image",help="Single-image prediction instead of dataset evaluation")
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    args=parser.parse_args()
    if args.batch_size<1 or not 0<args.fraction<=1:
        parser.error("Invalid batch size or sample fraction")
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else torch.device(args.device)
    checkpoint=ROOT / args.checkpoint if args.checkpoint else latest_completed_run(ROOT / "outputs") / "best.pt"
    model,state=load_checkpoint(checkpoint,device)
    print(f"CHECKPOINT: {checkpoint} | device: {device}",flush=True)
    size=state["config"]["image_size"]
    if args.image:
        path=ROOT / args.image
        loader=DataLoader(FundusDataset([{"path":str(path),"label":0}],size),batch_size=1)
        _,_,probabilities=predict_batches(model,loader,device)
        probability=float(probabilities[0])
        out=create_run_directory(ROOT / "outputs/prediction")
        result=dict(image=str(path),checkpoint=str(checkpoint),disease_probability=probability,
                    prediction="disease" if probability>=.5 else "normal",threshold=.5)
        write_json(out / "prediction.json",result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        print(f"RESULTS: {out}")
        return
    seed=args.seed if args.seed is not None else secrets.randbits(32)
    test_dir=resolve_test_directory(ROOT,state["config"].get("test_dir","data/2-MedImage-TestSet"),args.test_dir)
    rows,images=load_images(test_dir,size)
    selected=select_subset(rows,args.fraction,seed)
    out=create_run_directory(ROOT / "outputs/test80")
    details=dict(checkpoint=str(checkpoint),checkpoint_epoch=state.get("epoch"),seed=seed,
                 fraction=args.fraction,total_images=len(rows),sampled_images=len(selected),
                 normal=sum(r["label"]==0 for r in selected),disease=sum(r["label"]==1 for r in selected),
                 test_dir=str(test_dir),device=str(device),torch=torch.__version__)
    write_json(out / "evaluation_config.json",details)
    write_json(out / "sample_manifest.json",selected)
    loader=DataLoader(FundusDataset(selected,size,images=images),batch_size=args.batch_size)
    loss,labels,probabilities=predict_batches(model,loader,device)
    metrics=classification_metrics(labels,probabilities)
    metrics.update(unweighted_bce_loss=loss,loss_definition="unweighted_mean_bce",seed=seed,sampled_images=len(selected))
    summary=save_results(out,metrics,selected,probabilities,details=details,title="测试集评估结果")
    print(summary,flush=True)
    print(f"RESULTS: {out}",flush=True)


if __name__=="__main__":
    main()
