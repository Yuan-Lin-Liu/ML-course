"""Standalone training script: uv run python train.py --epochs 40."""
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

from sklearn.model_selection import train_test_split

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

def split_data(rows, fraction, seed):
    # Group exact pixel duplicates so a repeated image cannot cross partitions.
    groups={}
    for row in rows:
        key=row['sha256_pixels']
        if key in groups and groups[key][0]['label']!=row['label']:
            raise ValueError('Identical image has conflicting labels')
        groups.setdefault(key,[]).append(row)
    keys=list(groups)
    a,b=train_test_split(keys,test_size=fraction,random_state=seed,
                         stratify=[groups[k][0]['label'] for k in keys])
    return [row for k in a for row in groups[k]],[row for k in b for row in groups[k]]

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

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Training/validation scalars only; do not retain discarded predictions."""
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images).flatten()
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            count += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += ((logits.detach() >= 0) == labels.bool()).sum().item()
    return loss_sum / count, correct / count

def save_curves(history,out):
    for metric,label in [('loss','Weighted BCE loss'),('accuracy','Accuracy')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for split in ['train','val']:
            ax.plot([r['epoch'] for r in history],[r[f'{split}_{metric}'] for r in history],label=split)
        ax.set(xlabel='Epoch',ylabel=label,title=f'Training and validation {metric}')
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{metric}.png',dpi=160); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 and save model, validation results and curves.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--patience", type=int, default=0, help="0: run all epochs; >0: early stopping")
    parser.add_argument("--train-dir", default="data/2-MedImage-TrainSet")
    parser.add_argument("--test-dir", default="data/2-MedImage-TestSet", help="Used only to check train/test duplicates, never to tune the model")
    parser.add_argument("--output", default="outputs/resnet18")
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs,args.batch_size,args.image_size) < 1 or args.lr <= 0 or args.weight_decay < 0 or args.patience < 0 or not 0 < args.val_fraction < 1:
        parser.error("Invalid training parameters")
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = create_run_directory(ROOT / args.output)
    config = vars(args).copy()
    config.update(learning_rate=args.lr, validation_fraction=args.val_fraction, output=str(out))
    write_json(out / "config.json", config)
    print(f"RUN_DIRECTORY: {out} | device: {device}",flush=True)
    rows, images = load_images(ROOT / args.train_dir, args.image_size)
    test_rows, test_images = load_images(ROOT / args.test_dir, args.image_size)
    del test_images
    if len(rows) != 1639 or len(test_rows) != 250:
        raise ValueError("Expected 1639 training and 250 test images")
    if {r["sha256_pixels"] for r in rows} & {r["sha256_pixels"] for r in test_rows}:
        raise ValueError("Exact train/test duplicate images detected")
    train, val = split_data(rows,args.val_fraction,args.seed)
    write_json(out / "split_manifest.json", {"train":train,"val":val,"test":test_rows})
    environment = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                       device=torch.cuda.get_device_name(0) if device.type=="cuda" else "cpu",
                       counts={"train":len(train),"val":len(val),"test":len(test_rows)})
    write_json(out / "environment.json", environment)
    train_loader = DataLoader(FundusDataset(train,args.image_size,True,images),batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(FundusDataset(val,args.image_size,images=images),batch_size=args.batch_size)
    model = build_model().to(device)
    positives = sum(row["label"] for row in train)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train)-positives)/positives,device=device))
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    history, best_loss, stale = [], float("inf"), 0
    for epoch in range(1,args.epochs+1):
        train_loss, train_accuracy = run_epoch(model,train_loader,criterion,device,optimizer,scaler)
        val_loss, val_accuracy = run_epoch(model,val_loader,criterion,device)
        row = dict(epoch=epoch,train_loss=train_loss,train_accuracy=train_accuracy,
                   val_loss=val_loss,val_accuracy=val_accuracy,lr=optimizer.param_groups[0]["lr"])
        history.append(row)
        with (out / "history.csv").open("a",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=list(row))
            if epoch==1:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row),flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save(dict(model=model.state_dict(),epoch=epoch,val_loss=val_loss,
                            config=config,class_map={"normal":0,"disease":1}),out / "best.pt")
        else:
            stale += 1
        scheduler.step()
        if args.patience and stale >= args.patience:
            break
    write_json(out / "history.json",history)
    save_curves(history,out)
    best = min(history,key=lambda row:row["val_loss"])
    summary = dict(epochs_completed=len(history),best_epoch=best["epoch"],
                   best_val_loss=best["val_loss"],best_val_accuracy=best["val_accuracy"],
                   final_train_accuracy=history[-1]["train_accuracy"],
                   final_val_accuracy=history[-1]["val_accuracy"],
                   note="Training/validation only. Run evaluate.py for test metrics.")
    page='<!doctype html><meta charset="utf-8"><title>训练结果</title><style>body{max-width:900px;margin:30px auto;font:16px/1.8 sans-serif}img{max-width:100%}</style>'
    page+='<h1>训练与验证结果</h1><pre>'+html.escape(json.dumps(summary,indent=2,ensure_ascii=False))+'</pre>'
    page+='<img src="loss.png"><img src="accuracy.png">'
    (out / "results.html").write_text(page,encoding="utf-8")
    # Completion marker: only written when weights, history and curves are ready.
    write_json(out / "metrics.json",summary)
    print("TRAINING_COMPLETE " + json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
