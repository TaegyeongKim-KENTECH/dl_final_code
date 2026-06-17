import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from clipfordetectiondata.datasets import TestDataset1
from models.clipnet import OpenClipLinear
from models.clipnet_dyn import DynFakeDetector

_ROOT = Path(__file__).resolve().parents[1]

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate baseline or dyn detector on OOD test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "dyn"],
        required=True,
        help="baseline: OpenClipLinear | dyn: DynFakeDetector",
    )
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--test-path", type=str, default=str(_ROOT / "testset"))
    p.add_argument("--weights", type=str, default=str(_ROOT / "weights" / "open_clip_pytorch_model.bin"))
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth checkpoint")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--next-to-last", action="store_true", help="Remove CLIP projection layer")

    dyn = p.add_argument_group("dyn-only options")
    dyn.add_argument("--conf-threshold", type=float, default=0.8)
    dyn.add_argument("--temp", type=float, default=1.0)
    dyn.add_argument(
        "--sweep",
        action="store_true",
        help="Run confidence threshold sweep (dyn mode only)",
    )

    return p.parse_args(argv)

def build_model(args, device):
    if args.mode == "baseline":
        model = OpenClipLinear(
            normalize=True,
            next_to_last=args.next_to_last,
            pretrained_model_path=args.weights,
        )
    else:
        model = DynFakeDetector(
            pretrained_model_path=args.weights,
            normalize=True,
            next_to_last=args.next_to_last,
            conf_threshold=args.conf_threshold,
            temp=args.temp,
        )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_baseline(model, dataloader, device):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    predictions, labels, probabilities = [], [], []
    total_loss = 0.0
    folder_accuracies = {}
    folder_probabilities = {}
    folder_labels = {}
    folder_predictions = {}

    with torch.no_grad():
        for _, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Evaluating"
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            batch_probs = torch.sigmoid(outputs).cpu().numpy()
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
                    folder_labels[folder_name] = []
                    folder_predictions[folder_name] = []
                folder_probabilities[folder_name].append(batch_probs[i])
                folder_labels[folder_name].append(targets[i].item())
                folder_predictions[folder_name].append(predicted[i].item())

    return _summarize(
        predictions, labels, probabilities, total_loss, len(dataloader),
        folder_accuracies, folder_probabilities, folder_labels, folder_predictions,
        semantic_ratio=None,
    )

def evaluate_dyn(model, dataloader, device, conf_threshold=None):
    if conf_threshold is not None:
        model.conf_threshold = conf_threshold

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    predictions, labels, probabilities = [], [], []
    total_loss = 0.0
    total_samples = 0
    artifact_only = 0
    folder_accuracies = {}
    folder_probabilities = {}
    folder_labels = {}
    folder_predictions = {}

    with torch.no_grad():
        for _, (inputs, targets, folder_names) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Evaluating"
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            artifact_feat = model.artifact_branch.encode(inputs)
            pred_artifact = model.artifact_branch.classifier(artifact_feat)
            prob_art = torch.sigmoid(pred_artifact)
            confidence = (prob_art - 0.5).abs() * 2

            if (confidence >= model.conf_threshold).all():
                outputs = pred_artifact
                artifact_only += targets.size(0)
            else:
                pred_semantic = model.semantic_branch(inputs)
                gate_logits = model.gate(artifact_feat)
                weight = torch.softmax(gate_logits / model.temp, dim=-1)
                outputs = weight[:, 0:1] * pred_artifact + weight[:, 1:2] * pred_semantic

            outputs = outputs.squeeze()
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()
            total_samples += targets.size(0)

            predicted = (outputs > 0.5).float()
            batch_probs = torch.sigmoid(outputs).cpu().numpy()
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
                    folder_labels[folder_name] = []
                    folder_predictions[folder_name] = []
                folder_probabilities[folder_name].append(batch_probs[i])
                folder_labels[folder_name].append(targets[i].item())
                folder_predictions[folder_name].append(predicted[i].item())

    semantic_ratio = 1.0 - artifact_only / max(total_samples, 1)
    return _summarize(
        predictions, labels, probabilities, total_loss, len(dataloader),
        folder_accuracies, folder_probabilities, folder_labels, folder_predictions,
        semantic_ratio=semantic_ratio,
    )

def _summarize(predictions, labels, probabilities, total_loss, n_batches,
               folder_accuracies, folder_probabilities, folder_labels, folder_predictions,
               semantic_ratio):
    accuracy = accuracy_score(labels, predictions)
    average_precision = average_precision_score(labels, probabilities)
    avg_loss = total_loss / n_batches

    folder_aps = {}
    folder_total_accuracies = {}
    for folder_name in folder_probabilities:
        fp = folder_probabilities[folder_name]
        fl = folder_labels[folder_name]
        fprd = folder_predictions[folder_name]
        try:
            ap = average_precision_score(fl, fp)
        except ValueError:
            ap = 0.0
        total_correct = sum(1 for l, p in zip(fl, fprd) if l == p)
        folder_aps[folder_name] = ap
        folder_total_accuracies[folder_name] = total_correct / len(fl)

    return (
        accuracy, folder_accuracies, avg_loss,
        average_precision, folder_aps, folder_total_accuracies,
        semantic_ratio,
    )

def print_results(results, mode, conf_threshold=None):
    (
        accuracy, folder_accuracies, loss,
        average_precision, folder_aps, folder_total_accuracies,
        semantic_ratio,
    ) = results

    print(f"\nMode                         : {mode}")
    print(f"Overall Test Loss            : {loss:.4f}")
    print(f"Overall Test Accuracy        : {accuracy:.4f}")
    print(f"Overall Test Average Precision: {average_precision:.4f}")
    if semantic_ratio is not None:
        print(f"Semantic branch called       : {semantic_ratio * 100:.1f}% of samples")
        print(f"(conf_threshold = {conf_threshold})")
    print("-" * 70)

    for folder_name, acc in folder_accuracies.items():
        acc0 = acc["correct_0"] / acc["total_0"] if acc["total_0"] > 0 else 0.0
        acc1 = acc["correct_1"] / acc["total_1"] if acc["total_1"] > 0 else 0.0
        ap = folder_aps.get(folder_name, 0.0)
        total = folder_total_accuracies.get(folder_name, 0.0)
        print(
            f"Folder: {folder_name:30s} | "
            f"Acc(real)={acc0:.4f}  Acc(fake)={acc1:.4f}  "
            f"Total={total:.4f}  AP={ap:.4f}"
        )

def run_sweep(model, dataloader, device, save_dir, mode):
    print("\n" + "=" * 70)
    print("Threshold sweep")
    print("=" * 70)

    sweep_results = []
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        results = evaluate_dyn(model, dataloader, device, conf_threshold=thr)
        acc, _, _, ap, _, _, sem_ratio = results
        print(
            f"  conf_threshold={thr:.1f} | "
            f"Acc={acc:.4f}  AP={ap:.4f}  "
            f"semantic_called={sem_ratio * 100:.1f}%"
        )
        sweep_results.append({
            "conf_threshold": thr,
            "accuracy": round(acc, 6),
            "average_precision": round(ap, 6),
            "semantic_called": round(sem_ratio, 6),
        })

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"sweep_results_{mode}.json")
    csv_path = os.path.join(save_dir, f"sweep_results_{mode}.csv")

    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["conf_threshold", "accuracy", "average_precision", "semantic_called"],
        )
        writer.writeheader()
        writer.writerows(sweep_results)

    print(f"\nSweep results saved → {json_path}")
    print(f"Sweep results saved → {csv_path}")

def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = TestDataset1(
        is_train=False,
        args={"data_path": args.test_path, "eval_data_path": args.test_path},
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, device)

    if args.mode == "baseline":
        results = evaluate_baseline(model, test_dataloader, device)
    else:
        results = evaluate_dyn(model, test_dataloader, device, conf_threshold=args.conf_threshold)

    print_results(results, args.mode, conf_threshold=getattr(model, "conf_threshold", None))

    if args.sweep:
        if args.mode != "dyn":
            print("Warning: --sweep is only supported for --mode dyn. Skipping.")
        else:
            save_dir = str(Path(args.checkpoint).parent)
            run_sweep(model, test_dataloader, device, save_dir, args.mode)

if __name__ == "__main__":
    main()
