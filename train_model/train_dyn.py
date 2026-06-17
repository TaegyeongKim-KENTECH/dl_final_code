"""
train_dyn.py  ―  DynFakeDetector 학습 / 평가 스크립트
=======================================================

학습 루프 핵심 변경점
---------------------
  model(data) → (out, gate_reg_loss) = model(data)   [training=True]
  loss = BCE(out, target) + lossw * gate_reg_loss

  gate_reg_loss = -log(w_selected_path).mean()
    → 선택된 branch의 gate confidence를 높이는 방향으로 gate 학습
    → lossw 가 크면 gate가 더 decisive해짐 (artifact만으로 판단하려 함)

평가 루프
---------
  model.eval() 상태에서 model(data) → logit [B,1] 만 반환 (early exit 포함)
  evaluate_model() 내에서 semantic branch 호출 비율을 자동 추적
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from clipfordetectiondata.datasets import TrainDataset, TestDataset
from models.clipnet_dyn import DynFakeDetector


# ============================================================================
# 평가 함수
# ============================================================================

def evaluate_model(model, dataloader, device, writer, prefix, epoch):
    """
    eval 모드에서 early exit 포함 추론.
    semantic branch가 몇 % 샘플에 실제로 호출됐는지 추적.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    predictions, labels = [], []
    total_loss       = 0.0
    total_samples    = 0
    # early-exit 추적: artifact branch 만으로 처리된 샘플 수
    artifact_only    = 0
    folder_stats     = defaultdict(lambda: {"predictions": [], "labels": []})

    # confidence 기록용 hook
    _conf_log = []

    def _infer_with_tracking(x):
        """
        model._forward_infer 와 동일하되 early-exit 여부를 기록.
        """
        nonlocal artifact_only

        if model.infer_mode == 1:
            return model.artifact_branch(x)
        if model.infer_mode == 2:
            return model.semantic_branch(x)

        # artifact branch
        artifact_feat = model.artifact_branch.encode(x)
        pred_artifact = model.artifact_branch.classifier(artifact_feat)

        prob       = torch.sigmoid(pred_artifact)
        confidence = (prob - 0.5).abs() * 2
        _conf_log.append(confidence.detach().cpu())

        if (confidence >= model.conf_threshold).all():
            artifact_only += pred_artifact.size(0)
            return pred_artifact

        # semantic branch (불확실 샘플 존재)
        pred_semantic = model.semantic_branch(x)

        if model.infer_mode == -1:
            w = torch.full((artifact_feat.size(0), 2), 0.5,
                           device=artifact_feat.device)
        else:
            gate_logits = model.gate(artifact_feat)
            w = torch.softmax(gate_logits / model.temp, dim=-1)

        return w[:, 0:1] * pred_artifact + w[:, 1:2] * pred_semantic

    with torch.no_grad():
        for batch_idx, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader), total=len(dataloader),
            desc=f"Evaluating Epoch {epoch + 1}",
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = _infer_with_tracking(inputs).squeeze()

            loss = criterion(outputs, targets.float())
            total_loss    += loss.item()
            total_samples += targets.size(0)

            predicted = (outputs > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

            for fn, p, l in zip(
                folder_names,
                predicted.cpu().numpy(),
                targets.cpu().numpy(),
            ):
                folder_stats[fn]["predictions"].append(p)
                folder_stats[fn]["labels"].append(l)

    ac   = accuracy_score(labels, predictions)
    loss = total_loss / len(dataloader)

    # early-exit 통계
    semantic_call_ratio = 1.0 - artifact_only / max(total_samples, 1)
    avg_conf = torch.cat(_conf_log).mean().item() if _conf_log else 0.0

    writer.add_scalar(f"{prefix}/loss",               loss,               epoch)
    writer.add_scalar(f"{prefix}/accuracy",           ac,                 epoch)
    writer.add_scalar(f"{prefix}/semantic_call_ratio", semantic_call_ratio, epoch)
    writer.add_scalar(f"{prefix}/avg_confidence",     avg_conf,           epoch)

    print(
        f"  [Eval] AC={ac:.4f}  loss={loss:.4f}  "
        f"semantic_called={semantic_call_ratio * 100:.1f}%  "
        f"avg_conf={avg_conf:.4f}"
    )

    # 폴더별 상세
    for fn, stats in folder_stats.items():
        fp, fl = stats["predictions"], stats["labels"]
        fac  = accuracy_score(fl, fp)
        f0p  = [p for p, l in zip(fp, fl) if l == 0]
        f1p  = [p for p, l in zip(fp, fl) if l == 1]
        f0ac = accuracy_score([0] * len(f0p), f0p) if f0p else 0.0
        f1ac = accuracy_score([1] * len(f1p), f1p) if f1p else 0.0
        print(f"    {fn}: AC={fac:.4f} | real={f0ac:.4f} | fake={f1ac:.4f}")

    return ac, -ac   # EarlyStopping은 val_loss 기준


# ============================================================================
# EarlyStopping
# ============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False):
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
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
            print(
                f"  Val improved ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving..."
            )
        torch.save(model.state_dict(), "checkpoint.pth")
        self.val_loss_min = val_loss


# ============================================================================
# 학습 함수
# ============================================================================

def train_model(
    model,
    train_dataloader,
    test_dataloader,
    epochs:       int,
    device:       torch.device,
    save_path:    str,
    lossw:        float = 0.01,
    lr:           float = 1e-4,
    weight_decay: float = 0.0,
):
    """
    Parameters
    ----------
    lossw : gate_reg_loss 가중치.
        - 높이면 gate가 더 decisive → artifact branch를 더 자주 사용
        - 0으로 설정하면 gate reg 없이 task loss만으로 학습
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_ac          = 0.0
    best_model_state = None
    early_stopping   = EarlyStopping(patience=5, verbose=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = SummaryWriter(os.path.join(save_path, f"train_log{current_time}"))
    eval_writer  = SummaryWriter(os.path.join(save_path, f"eval_log{current_time}"))
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_bce   = 0.0
        running_gate  = 0.0
        path_counts   = [0, 0]   # [artifact 선택 횟수, semantic 선택 횟수]
        total_batches = len(train_dataloader)

        for batch_idx, (data, target) in tqdm(
            enumerate(train_dataloader),
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{epochs}",
        ):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # ── DynMM 학습: (logit, gate_reg_loss) 반환 ──────────────────
            out, gate_reg_loss = model(data)

            loss_bce  = criterion(out.squeeze(), target.float())
            loss      = loss_bce + lossw * gate_reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()

            running_total += loss.item()
            running_bce   += loss_bce.item()
            running_gate  += gate_reg_loss.item()

            # path 통계: randint(0,2)가 어떤 쪽을 선택했는지
            # (model 내부에서 직접 노출하기 위해 simple proxy 사용)
            # gate_reg_loss가 작으면 artifact branch가 자신있게 선택된 것
            # 정확한 통계는 아래처럼 gate weight를 직접 조회
            with torch.no_grad():
                model.eval()
                af = model.artifact_branch.encode(data)
                gw = torch.softmax(
                    model.gate(af) / model.temp, dim=-1
                )                              # [B, 2]
                path_counts[0] += (gw[:, 0] > gw[:, 1]).sum().item()
                path_counts[1] += (gw[:, 1] >= gw[:, 0]).sum().item()
                model.train()

        # epoch 통계
        ep_total = running_total / total_batches
        ep_bce   = running_bce   / total_batches
        ep_gate  = running_gate  / total_batches
        art_ratio = path_counts[0] / max(sum(path_counts), 1)

        print(
            f"Epoch {epoch + 1} | "
            f"total={ep_total:.4f}  bce={ep_bce:.4f}  gate_reg={ep_gate:.4f}  "
            f"artifact_preferred={art_ratio * 100:.1f}%"
        )
        train_writer.add_scalar("loss/total",            ep_total, epoch)
        train_writer.add_scalar("loss/bce",              ep_bce,   epoch)
        train_writer.add_scalar("loss/gate_reg",         ep_gate,  epoch)
        train_writer.add_scalar("gate/artifact_ratio",  art_ratio, epoch)

        # ── 평가 ──────────────────────────────────────────────────────────
        ac, val_loss = evaluate_model(
            model, test_dataloader, device, eval_writer, "validation", epoch
        )

        # ── 모델 저장 ──────────────────────────────────────────────────────
        torch.save(
            model.state_dict(),
            os.path.join(save_path, f"model_epoch_{epoch + 1}_{current_time}.pth"),
        )

        if ac > best_ac:
            best_ac          = ac
            best_model_state = model.state_dict()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training finished.")
    train_writer.close()
    eval_writer.close()

    if best_model_state:
        model.load_state_dict(best_model_state)
        best_path = os.path.join(save_path, f"best_model_{current_time}.pth")
        torch.save(model.state_dict(), best_path)
        print(f"Best AC: {best_ac:.4f}  →  {best_path}")
    else:
        print("No improved model found.")


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        "DynFakeDetector training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu",            type=str,   default="1")
    p.add_argument("--data-path",      type=str,   default="/home/work/ktg0829/final_project/clipforfakedetection/trainset/")
    p.add_argument("--val-path",       type=str,   default="/home/work/ktg0829/final_project/clipforfakedetection/valset/")
    p.add_argument("--weights",        type=str,   default="/home/work/ktg0829/final_project/clipforfakedetection/weights/open_clip_pytorch_model.bin")
    p.add_argument("--save-path",      type=str,   default="/home/work/ktg0829/final_project/clipforfakedetection/weights/model_save/")
    p.add_argument("--epochs",         type=int,   default=5)
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=0.0)
    p.add_argument(
        "--lossw", type=float, default=0.01,
        help="gate regularization loss weight. 높을수록 artifact branch 선호.",
    )
    p.add_argument(
        "--temp", type=float, default=1.0,
        help="DiffSoftmax 온도. 낮을수록 hard 선택.",
    )
    p.add_argument("--hard-gate",    action="store_true", help="straight-through hard gate")
    p.add_argument("--freeze-clip",  action="store_true", help="CLIP backbone freeze")
    p.add_argument("--next-to-last", action="store_true", help="CLIP proj 제거")
    p.add_argument(
        "--conf-threshold", type=float, default=0.8,
        help="Early-exit confidence threshold. 이 값 이상이면 semantic branch 생략.",
    )
    return p.parse_args()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 데이터 ───────────────────────────────────────────────────────────
    train_dataset = TrainDataset(
        is_train=True, args={"data_path": args.data_path}
    )
    test_dataset = TestDataset(
        is_train=False,
        args={"data_path": args.val_path, "eval_data_path": args.val_path},
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── 모델 ─────────────────────────────────────────────────────────────
    model = DynFakeDetector(
        pretrained_model_path=args.weights,
        temp=args.temp,
        hard_gate=args.hard_gate,
        freeze_clip=args.freeze_clip,
        next_to_last=args.next_to_last,
        conf_threshold=args.conf_threshold,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable / 1e6:.2f}M")

    # ── 학습 ─────────────────────────────────────────────────────────────
    train_model(
        model,
        train_dataloader,
        test_dataloader,
        epochs=args.epochs,
        device=device,
        save_path=args.save_path,
        lossw=args.lossw,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
