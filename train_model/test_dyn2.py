import os
import csv
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

from clipfordetectiondata.datasets import TestDataset1
from models.clipnet_dyn import DynFakeDetector

# 사용할 물리 GPU 번호 지정 (서버에서 쓰고 싶은 GPU 번호로 변경)
# CUDA_VISIBLE_DEVICES 설정 이후에는 항상 논리적으로 cuda:0 이 되므로
# device는 반드시 "cuda:0" 으로 고정해야 함
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ============================================================================
# 평가 함수
# ============================================================================

def evaluate_model(model, dataloader, device, conf_threshold=None):
    """
    Parameters
    ----------
    conf_threshold : float | None
        None 이면 model.conf_threshold 그대로 사용.
        값을 넘기면 그 값으로 덮어씀 (threshold sweep 실험용).
    """
    if conf_threshold is not None:
        model.conf_threshold = conf_threshold

    model.eval()

    predictions, labels, probabilities = [], [], []
    total_loss    = 0.0
    total_samples = 0
    artifact_only = 0   # early-exit 으로 artifact만 사용된 샘플 수

    folder_accuracies    = {}
    folder_probabilities = {}
    folder_labels        = {}
    folder_predictions   = {}
    folder_semantic      = {}  # 폴더별 {semantic_called, total} 카운터

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Evaluating"
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            # ── Early-exit 여부를 직접 추적하는 infer ────────────────────
            artifact_feat = model.artifact_branch.encode(inputs)
            pred_artifact = model.artifact_branch.classifier(artifact_feat)

            prob_art    = torch.sigmoid(pred_artifact)
            confidence  = (prob_art - 0.5).abs() * 2          # [B,1] ∈ [0,1]

            # 샘플별로 semantic 호출 여부 기록 ([B] bool 텐서)
            needs_semantic = (confidence.squeeze(1) < model.conf_threshold)  # [B]

            if not needs_semantic.any():
                # 배치 전체 early-exit
                outputs = pred_artifact
                artifact_only += targets.size(0)
            else:
                # Semantic branch 실행 후 gate 가중합
                pred_semantic = model.semantic_branch(inputs)
                gate_logits   = model.gate(artifact_feat)
                weight        = torch.softmax(
                    gate_logits / model.temp, dim=-1
                )                                              # [B,2]
                outputs = (weight[:, 0:1] * pred_artifact
                         + weight[:, 1:2] * pred_semantic)
                # early-exit된 샘플 수만큼 차감
                artifact_only += (~needs_semantic).sum().item()

            outputs = outputs.squeeze()
            loss     = criterion(outputs, targets.float())
            total_loss    += loss.item()
            total_samples += targets.size(0)

            predicted         = (outputs > 0.5).float()
            batch_probs       = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            probabilities.extend(batch_probs)

            # 폴더별 통계 수집
            for i, folder_name in enumerate(folder_names):
                # accuracy_0 / accuracy_1 카운터
                if folder_name not in folder_accuracies:
                    folder_accuracies[folder_name] = {
                        "correct_0": 0, "total_0": 0,
                        "correct_1": 0, "total_1": 0,
                    }
                if targets[i].item() == 0:
                    folder_accuracies[folder_name]["total_0"] += 1
                    if predicted[i].item() == 0:
                        folder_accuracies[folder_name]["correct_0"] += 1
                else:
                    folder_accuracies[folder_name]["total_1"] += 1
                    if predicted[i].item() == 1:
                        folder_accuracies[folder_name]["correct_1"] += 1

                # AP / total accuracy 계산용
                if folder_name not in folder_probabilities:
                    folder_probabilities[folder_name] = []
                    folder_labels[folder_name]        = []
                    folder_predictions[folder_name]   = []
                folder_probabilities[folder_name].append(batch_probs[i])
                folder_labels[folder_name].append(targets[i].item())
                folder_predictions[folder_name].append(predicted[i].item())

                # 폴더별 semantic 호출 카운터
                if folder_name not in folder_semantic:
                    folder_semantic[folder_name] = {"semantic_called": 0, "total": 0}
                folder_semantic[folder_name]["total"] += 1
                if needs_semantic[i].item():
                    folder_semantic[folder_name]["semantic_called"] += 1

    accuracy          = accuracy_score(labels, predictions)
    average_precision = average_precision_score(labels, probabilities)
    avg_loss          = total_loss / len(dataloader)
    semantic_ratio    = 1.0 - artifact_only / max(total_samples, 1)

    # 폴더별 AP / Total Accuracy
    folder_aps              = {}
    folder_total_accuracies = {}
    for folder_name in folder_probabilities:
        fp   = folder_probabilities[folder_name]
        fl   = folder_labels[folder_name]
        fprd = folder_predictions[folder_name]
        try:
            ap = average_precision_score(fl, fp)
        except ValueError:
            ap = 0.0
        total_correct  = sum(1 for l, p in zip(fl, fprd) if l == p)
        folder_aps[folder_name]              = ap
        folder_total_accuracies[folder_name] = total_correct / len(fl)

    return (
        accuracy, folder_accuracies, avg_loss,
        average_precision, folder_aps, folder_total_accuracies,
        semantic_ratio, folder_semantic,
    )


# ============================================================================
# 테스트 함수
# ============================================================================

def test_model(model, dataloader, device, conf_threshold=None, save_path=None):
    (
        accuracy, folder_accuracies, loss,
        average_precision, folder_aps, folder_total_accuracies,
        semantic_ratio, folder_semantic,
    ) = evaluate_model(model, dataloader, device, conf_threshold)

    print(f"\nOverall Test Loss            : {loss:.4f}")
    print(f"Overall Test Accuracy        : {accuracy:.4f}")
    print(f"Overall Test Average Precision: {average_precision:.4f}")
    print(f"Semantic branch called       : {semantic_ratio * 100:.1f}% of samples")
    print(f"(conf_threshold = {model.conf_threshold})")
    print("-" * 70)

    for folder_name, acc in folder_accuracies.items():
        acc0  = acc["correct_0"] / acc["total_0"] if acc["total_0"] > 0 else 0.0
        acc1  = acc["correct_1"] / acc["total_1"] if acc["total_1"] > 0 else 0.0
        ap    = folder_aps.get(folder_name, 0.0)
        total = folder_total_accuracies.get(folder_name, 0.0)
        sem   = folder_semantic.get(folder_name, {})
        sem_r = sem["semantic_called"] / sem["total"] if sem.get("total", 0) > 0 else 0.0
        print(
            f"Folder: {folder_name:30s} | "
            f"Acc(real)={acc0:.4f}  Acc(fake)={acc1:.4f}  "
            f"Total={total:.4f}  AP={ap:.4f}  "
            f"Semantic={sem_r * 100:.1f}%"
        )

    # ── 데이터셋별 성능 막대그래프 ────────────────────────────────────────
    if save_path is not None:
        _plot_per_dataset(
            folder_accuracies, folder_aps, folder_total_accuracies,
            folder_semantic, accuracy, average_precision, save_path,
        )


def _plot_per_dataset(
    folder_accuracies, folder_aps, folder_total_accuracies,
    folder_semantic, overall_acc, overall_ap, save_path,
):
    folders   = sorted(folder_accuracies.keys())
    acc_real  = [folder_accuracies[f]["correct_0"] / folder_accuracies[f]["total_0"]
                 if folder_accuracies[f]["total_0"] > 0 else 0.0 for f in folders]
    acc_fake  = [folder_accuracies[f]["correct_1"] / folder_accuracies[f]["total_1"]
                 if folder_accuracies[f]["total_1"] > 0 else 0.0 for f in folders]
    acc_total = [folder_total_accuracies.get(f, 0.0) for f in folders]
    aps       = [folder_aps.get(f, 0.0) for f in folders]
    sem_ratios = [
        folder_semantic[f]["semantic_called"] / folder_semantic[f]["total"]
        if folder_semantic.get(f, {}).get("total", 0) > 0 else 0.0
        for f in folders
    ]

    x      = np.arange(len(folders))
    width  = 0.25
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(folders) * 0.9), 14))
    fig.suptitle("Per-Dataset Performance", fontsize=14, fontweight="bold")

    # ── 1. Accuracy (real / fake / total) ────────────────────────────────
    ax = axes[0]
    ax.bar(x - width, acc_real,  width, label="Acc (real)",  color="steelblue",  alpha=0.85)
    ax.bar(x,         acc_fake,  width, label="Acc (fake)",  color="darkorange", alpha=0.85)
    ax.bar(x + width, acc_total, width, label="Acc (total)", color="seagreen",   alpha=0.85)
    ax.axhline(overall_acc, color="seagreen", linestyle="--", linewidth=1.2,
               label=f"Overall Acc ({overall_acc:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(folders, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # ── 2. Average Precision ─────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(x, aps, width * 1.5, color="mediumpurple", alpha=0.85)
    ax.axhline(overall_ap, color="mediumpurple", linestyle="--", linewidth=1.2,
               label=f"Overall AP ({overall_ap:.3f})")
    for bar, val in zip(bars, aps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(folders, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Average Precision")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # ── 3. Semantic Branch Called (%) ─────────────────────────────────────
    ax = axes[2]
    bars = ax.bar(x, [r * 100 for r in sem_ratios], width * 1.5,
                  color="salmon", alpha=0.85)
    overall_sem = np.mean(sem_ratios) * 100
    ax.axhline(overall_sem, color="salmon", linestyle="--", linewidth=1.2,
               label=f"Overall avg ({overall_sem:.1f}%)")
    for bar, val in zip(bars, sem_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(folders, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Semantic branch called (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(save_path, "per_dataset_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPer-dataset plot saved → {out_path}")


# ============================================================================
# 진입점
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 모델 로드 ────────────────────────────────────────────────────────
    model_save_path = "/home/work/ktg0829/final_project/clipforfakedetection/weights/model_save/best_model_20260603-074224.pth"

    model = DynFakeDetector(
        pretrained_model_path="/home/work/ktg0829/final_project/clipforfakedetection/weights/open_clip_pytorch_model.bin",
        normalize=True,
        next_to_last=False,   # 학습 당시와 동일하게 맞춰야 함 (proj 포함, dim=768)
        conf_threshold=0.8,   # 추론 시 early-exit 기준값 (아래서 재설정 가능)
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    # ── 데이터 ───────────────────────────────────────────────────────────
    test_dataset = TestDataset1(
        is_train=False,
        args={
            "data_path": "/home/work/ktg0829/final_project/clipforfakedetection/testset/",
            "eval_data_path": "/home/work/ktg0829/final_project/clipforfakedetection/testset/",
        },
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 그래프 저장 경로 = 모델 저장 폴더와 동일
    plot_save_path = os.path.dirname(model_save_path)

    # ── 기본 테스트 + 데이터셋별 그래프 ─────────────────────────────────
    test_model(model, test_dataloader, device, save_path=plot_save_path)

    # ── conf_threshold sweep + 결과 저장 ────────────────────────────────
    print("\n" + "=" * 70)
    print("Threshold sweep")
    print("=" * 70)

    sweep_results = []   # 그래프 그릴 때 사용할 리스트

    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        (acc, _, _, ap, _, _, sem_ratio, _) = evaluate_model(
            model, test_dataloader, device, conf_threshold=thr
        )
        print(
            f"  conf_threshold={thr:.1f} | "
            f"Acc={acc:.4f}  AP={ap:.4f}  "
            f"semantic_called={sem_ratio * 100:.1f}%"
        )
        sweep_results.append({
            "conf_threshold":  thr,
            "accuracy":        round(acc, 6),
            "average_precision": round(ap, 6),
            "semantic_called": round(sem_ratio, 6),   # 0~1 비율로 저장
        })

    # JSON 저장 (구조적 데이터, 나중에 불러오기 쉬움)
    json_path = os.path.join(os.path.dirname(model_save_path), "sweep_results2.json")
    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved → {json_path}")

    # CSV 저장 (pandas / matplotlib 에서 바로 읽기 편함)
    csv_path = os.path.join(os.path.dirname(model_save_path), "sweep_results2.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conf_threshold", "accuracy", "average_precision", "semantic_called"]
        )
        writer.writeheader()
        writer.writerows(sweep_results)
    print(f"Sweep results saved → {csv_path}")