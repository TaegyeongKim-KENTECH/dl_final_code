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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def evaluate_model(model, dataloader, device, conf_threshold=None):
    if conf_threshold is not None:
        model.conf_threshold = conf_threshold

    model.eval()

    predictions, labels, probabilities = [], [], []
    total_loss    = 0.0
    total_samples = 0
    artifact_only = 0

    folder_accuracies    = {}
    folder_probabilities = {}
    folder_labels        = {}
    folder_predictions   = {}
    folder_semantic      = {}

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Evaluating"
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            artifact_feat = model.artifact_branch.encode(inputs)
            pred_artifact = model.artifact_branch.classifier(artifact_feat)

            prob_art    = torch.sigmoid(pred_artifact)
            confidence  = (prob_art - 0.5).abs() * 2

            needs_semantic = (confidence.squeeze(1) < model.conf_threshold)

            if not needs_semantic.any():
                outputs = pred_artifact
                artifact_only += targets.size(0)
            else:
                pred_semantic = model.semantic_branch(inputs)
                gate_logits   = model.gate(artifact_feat)
                weight        = torch.softmax(
                    gate_logits / model.temp, dim=-1
                )
                outputs = (weight[:, 0:1] * pred_artifact
                         + weight[:, 1:2] * pred_semantic)
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

            for i, folder_name in enumerate(folder_names):
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

                if folder_name not in folder_probabilities:
                    folder_probabilities[folder_name] = []
                    folder_labels[folder_name]        = []
                    folder_predictions[folder_name]   = []
                folder_probabilities[folder_name].append(batch_probs[i])
                folder_labels[folder_name].append(targets[i].item())
                folder_predictions[folder_name].append(predicted[i].item())

                if folder_name not in folder_semantic:
                    folder_semantic[folder_name] = {"semantic_called": 0, "total": 0}
                folder_semantic[folder_name]["total"] += 1
                if needs_semantic[i].item():
                    folder_semantic[folder_name]["semantic_called"] += 1

    accuracy          = accuracy_score(labels, predictions)
    average_precision = average_precision_score(labels, probabilities)
    avg_loss          = total_loss / len(dataloader)
    semantic_ratio    = 1.0 - artifact_only / max(total_samples, 1)

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

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_save_path = "/home/work/ktg0829/final_project/clipforfakedetection/weights/model_save/best_model_20260603-074224.pth"

    model = DynFakeDetector(
        pretrained_model_path="/home/work/ktg0829/final_project/clipforfakedetection/weights/open_clip_pytorch_model.bin",
        normalize=True,
        next_to_last=False,
        conf_threshold=0.8,
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    test_dataset = TestDataset1(
        is_train=False,
        args={
            "data_path": "/home/work/ktg0829/final_project/clipforfakedetection/testset/",
            "eval_data_path": "/home/work/ktg0829/final_project/clipforfakedetection/testset/",
        },
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    plot_save_path = os.path.dirname(model_save_path)

    test_model(model, test_dataloader, device, save_path=plot_save_path)

    print("\n" + "=" * 70)
    print("Threshold sweep")
    print("=" * 70)

    sweep_results = []

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
            "semantic_called": round(sem_ratio, 6),
        })

    json_path = os.path.join(os.path.dirname(model_save_path), "sweep_results2.json")
    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved → {json_path}")

    csv_path = os.path.join(os.path.dirname(model_save_path), "sweep_results2.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conf_threshold", "accuracy", "average_precision", "semantic_called"]
        )
        writer.writeheader()
        writer.writerows(sweep_results)
    print(f"Sweep results saved → {csv_path}")
