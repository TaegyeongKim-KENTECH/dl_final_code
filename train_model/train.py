import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from clipfordetectiondata.datasets import TestDataset, TrainDataset
from models.clipnet import OpenClipLinear
from models.clipnet_dyn import DynFakeDetector

_ROOT = Path(__file__).resolve().parents[1]

class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f"  Val improved ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving...")
        torch.save(model.state_dict(), "checkpoint.pth")
        self.val_loss_min = val_loss

def evaluate_baseline(model, dataloader, device, writer, prefix, epoch):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    predictions, labels = [], []
    total_loss = 0.0
    folder_stats = defaultdict(lambda: {"predictions": [], "labels": []})

    with torch.no_grad():
        for _, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Evaluating Epoch {epoch + 1}",
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

            for fn, p, l in zip(folder_names, predicted.cpu().numpy(), targets.cpu().numpy()):
                folder_stats[fn]["predictions"].append(p)
                folder_stats[fn]["labels"].append(l)

    ac = accuracy_score(labels, predictions)
    loss = total_loss / len(dataloader)
    writer.add_scalar(f"{prefix}/loss", loss, epoch)
    writer.add_scalar(f"{prefix}/accuracy", ac, epoch)

    for fn, stats in folder_stats.items():
        fp, fl = stats["predictions"], stats["labels"]
        fac = accuracy_score(fl, fp)
        f0p = [p for p, l in zip(fp, fl) if l == 0]
        f1p = [p for p, l in zip(fp, fl) if l == 1]
        f0ac = accuracy_score([0] * len(f0p), f0p) if f0p else 0.0
        f1ac = accuracy_score([1] * len(f1p), f1p) if f1p else 0.0
        print(f"    {fn}: AC={fac:.4f} | real={f0ac:.4f} | fake={f1ac:.4f}")

    print(f"  [Eval] AC={ac:.4f}  loss={loss:.4f}")
    return ac, -ac

def train_baseline(model, train_loader, val_loader, args, device):
    from torch.utils.tensorboard import SummaryWriter

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.freeze_clip:
        model.freeze_clip()

    best_ac = 0.0
    best_state = None
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(args.save_path, f"train_log_{args.mode}_{current_time}"))
    eval_writer = SummaryWriter(os.path.join(args.save_path, f"eval_log_{args.mode}_{current_time}"))

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for _, (data, target) in tqdm(
            enumerate(train_loader),
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / total_batches
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        train_writer.add_scalar("training_loss", epoch_loss, epoch)

        ac, val_loss = evaluate_baseline(model, val_loader, device, eval_writer, "validation", epoch)

        ckpt = os.path.join(args.save_path, f"model_{args.mode}_epoch_{epoch + 1}_{current_time}.pth")
        torch.save(model.state_dict(), ckpt)

        if ac > best_ac:
            best_ac = ac
            best_state = model.state_dict()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    train_writer.close()
    eval_writer.close()

    if best_state:
        model.load_state_dict(best_state)
        best_path = os.path.join(args.save_path, f"best_model_{args.mode}_{current_time}.pth")
        torch.save(model.state_dict(), best_path)
        print(f"Best AC: {best_ac:.4f}  →  {best_path}")
    else:
        print("No improved model found.")

def evaluate_dyn(model, dataloader, device, writer, prefix, epoch):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    predictions, labels = [], []
    total_loss = 0.0
    total_samples = 0
    artifact_only = 0
    folder_stats = defaultdict(lambda: {"predictions": [], "labels": []})
    conf_log = []

    def infer_with_tracking(x):
        nonlocal artifact_only
        if model.infer_mode == 1:
            return model.artifact_branch(x)
        if model.infer_mode == 2:
            return model.semantic_branch(x)

        artifact_feat = model.artifact_branch.encode(x)
        pred_artifact = model.artifact_branch.classifier(artifact_feat)
        prob = torch.sigmoid(pred_artifact)
        confidence = (prob - 0.5).abs() * 2
        conf_log.append(confidence.detach().cpu())

        if (confidence >= model.conf_threshold).all():
            artifact_only += pred_artifact.size(0)
            return pred_artifact

        pred_semantic = model.semantic_branch(x)
        if model.infer_mode == -1:
            w = torch.full((artifact_feat.size(0), 2), 0.5, device=artifact_feat.device)
        else:
            gate_logits = model.gate(artifact_feat)
            w = torch.softmax(gate_logits / model.temp, dim=-1)
        return w[:, 0:1] * pred_artifact + w[:, 1:2] * pred_semantic

    with torch.no_grad():
        for _, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Evaluating Epoch {epoch + 1}",
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = infer_with_tracking(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()
            total_samples += targets.size(0)

            predicted = (outputs > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

            for fn, p, l in zip(folder_names, predicted.cpu().numpy(), targets.cpu().numpy()):
                folder_stats[fn]["predictions"].append(p)
                folder_stats[fn]["labels"].append(l)

    ac = accuracy_score(labels, predictions)
    loss = total_loss / len(dataloader)
    semantic_ratio = 1.0 - artifact_only / max(total_samples, 1)
    avg_conf = torch.cat(conf_log).mean().item() if conf_log else 0.0

    writer.add_scalar(f"{prefix}/loss", loss, epoch)
    writer.add_scalar(f"{prefix}/accuracy", ac, epoch)
    writer.add_scalar(f"{prefix}/semantic_call_ratio", semantic_ratio, epoch)
    writer.add_scalar(f"{prefix}/avg_confidence", avg_conf, epoch)

    print(
        f"  [Eval] AC={ac:.4f}  loss={loss:.4f}  "
        f"semantic_called={semantic_ratio * 100:.1f}%  avg_conf={avg_conf:.4f}"
    )

    for fn, stats in folder_stats.items():
        fp, fl = stats["predictions"], stats["labels"]
        fac = accuracy_score(fl, fp)
        f0p = [p for p, l in zip(fp, fl) if l == 0]
        f1p = [p for p, l in zip(fp, fl) if l == 1]
        f0ac = accuracy_score([0] * len(f0p), f0p) if f0p else 0.0
        f1ac = accuracy_score([1] * len(f1p), f1p) if f1p else 0.0
        print(f"    {fn}: AC={fac:.4f} | real={f0ac:.4f} | fake={f1ac:.4f}")

    return ac, -ac

def train_dyn(model, train_loader, val_loader, args, device):
    from torch.utils.tensorboard import SummaryWriter

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_ac = 0.0
    best_state = None
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(args.save_path, f"train_log_{args.mode}_{current_time}"))
    eval_writer = SummaryWriter(os.path.join(args.save_path, f"eval_log_{args.mode}_{current_time}"))

    for epoch in range(args.epochs):
        model.train()
        running_total = running_bce = running_gate = 0.0
        path_counts = [0, 0]
        total_batches = len(train_loader)

        for _, (data, target) in tqdm(
            enumerate(train_loader),
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            out, gate_reg_loss = model(data)
            loss_bce = criterion(out.squeeze(), target.float())
            loss = loss_bce + args.lossw * gate_reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()

            running_total += loss.item()
            running_bce += loss_bce.item()
            running_gate += gate_reg_loss.item()

            with torch.no_grad():
                model.eval()
                af = model.artifact_branch.encode(data)
                gw = torch.softmax(model.gate(af) / model.temp, dim=-1)
                path_counts[0] += (gw[:, 0] > gw[:, 1]).sum().item()
                path_counts[1] += (gw[:, 1] >= gw[:, 0]).sum().item()
                model.train()

        ep_total = running_total / total_batches
        ep_bce = running_bce / total_batches
        ep_gate = running_gate / total_batches
        art_ratio = path_counts[0] / max(sum(path_counts), 1)

        print(
            f"Epoch {epoch + 1} | total={ep_total:.4f}  bce={ep_bce:.4f}  "
            f"gate_reg={ep_gate:.4f}  artifact_preferred={art_ratio * 100:.1f}%"
        )
        train_writer.add_scalar("loss/total", ep_total, epoch)
        train_writer.add_scalar("loss/bce", ep_bce, epoch)
        train_writer.add_scalar("loss/gate_reg", ep_gate, epoch)
        train_writer.add_scalar("gate/artifact_ratio", art_ratio, epoch)

        ac, val_loss = evaluate_dyn(model, val_loader, device, eval_writer, "validation", epoch)

        ckpt = os.path.join(args.save_path, f"model_{args.mode}_epoch_{epoch + 1}_{current_time}.pth")
        torch.save(model.state_dict(), ckpt)

        if ac > best_ac:
            best_ac = ac
            best_state = model.state_dict()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    train_writer.close()
    eval_writer.close()

    if best_state:
        model.load_state_dict(best_state)
        best_path = os.path.join(args.save_path, f"best_model_{args.mode}_{current_time}.pth")
        torch.save(model.state_dict(), best_path)
        print(f"Best AC: {best_ac:.4f}  →  {best_path}")
    else:
        print("No improved model found.")

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train baseline (Dual-Path) or dyn (Gating) detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "dyn"],
        required=True,
        help="baseline: OpenClipLinear | dyn: DynFakeDetector (gating)",
    )
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--data-path", type=str, default=str(_ROOT / "trainset"))
    p.add_argument("--val-path", type=str, default=str(_ROOT / "valset"))
    p.add_argument("--weights", type=str, default=str(_ROOT / "weights" / "open_clip_pytorch_model.bin"))
    p.add_argument("--save-path", type=str, default=str(_ROOT / "weights" / "model_save"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--freeze-clip", action="store_true", help="Freeze CLIP backbone (baseline)")
    p.add_argument("--next-to-last", action="store_true", help="Remove CLIP projection layer")

    dyn = p.add_argument_group("dyn-only options")
    dyn.add_argument("--lossw", type=float, default=0.01, help="Gate regularization weight")
    dyn.add_argument("--temp", type=float, default=1.0, help="DiffSoftmax temperature")
    dyn.add_argument("--hard-gate", action="store_true")
    dyn.add_argument("--conf-threshold", type=float, default=0.8, help="Early-exit threshold")

    return p.parse_args(argv)

def build_model(args):
    if args.mode == "baseline":
        model = OpenClipLinear(
            normalize=True,
            next_to_last=args.next_to_last,
            pretrained_model_path=args.weights,
        )
    else:
        model = DynFakeDetector(
            pretrained_model_path=args.weights,
            temp=args.temp,
            hard_gate=args.hard_gate,
            freeze_clip=args.freeze_clip,
            next_to_last=args.next_to_last,
            conf_threshold=args.conf_threshold,
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mode: {args.mode} | Trainable params: {trainable / 1e6:.2f}M")
    return model

def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TrainDataset(is_train=True, args={"data_path": args.data_path})
    val_dataset = TestDataset(
        is_train=False,
        args={"data_path": args.val_path, "eval_data_path": args.val_path},
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args)

    if args.mode == "baseline":
        train_baseline(model, train_loader, val_loader, args, device)
    else:
        train_dyn(model, train_loader, val_loader, args, device)

if __name__ == "__main__":
    main()
