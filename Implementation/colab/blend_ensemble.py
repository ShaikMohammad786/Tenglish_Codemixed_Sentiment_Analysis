#!/usr/bin/env python3
"""
Late-Fusion Probability Blending for CMTE-NITANP Sentiment Analysis.
Combines test probability predictions from multiple trained models to reach ~88%+ accuracy.
"""

import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def main():
    parser = argparse.ArgumentParser(description="Ensemble Probability Blending")
    parser.add_argument("--preds", nargs="+", required=True,
                        help="Paths to .npz prediction files (e.g. electra.npz deberta.npz)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Weights for each model (defaults to equal weighting)")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Temperature scaling factor for transformer probabilities")
    args = parser.parse_args()

    files = args.preds
    if args.weights:
        weights = np.array(args.weights, dtype=np.float32)
        weights /= weights.sum()
    else:
        weights = np.ones(len(files), dtype=np.float32) / len(files)

    print(f"[Ensemble] Loading {len(files)} prediction vectors...")
    prob_list = []
    labels = None

    for i, path in enumerate(files):
        data = np.load(path)
        p = data["probs"]
        y = data["labels"] if "labels" in data else data["targets"]
        if labels is None:
            labels = y
        else:
            assert np.array_equal(labels, y), "Ground truth labels mismatch between files!"

        # Apply temperature scaling to soften overconfidence
        if args.temperature != 1.0:
            p = np.exp(np.log(np.clip(p, 1e-7, 1.0)) / args.temperature)
            p /= p.sum(axis=1, keepdims=True)

        acc = accuracy_score(labels, p.argmax(axis=1))
        f1 = f1_score(labels, p.argmax(axis=1), average="macro")
        print(f"  Model {i+1} [{path}]: Acc = {acc*100:.2f}%, Macro-F1 = {f1*100:.2f}% (weight={weights[i]:.2f})")
        prob_list.append(p)

    # Weighted blending
    blended_probs = np.zeros_like(prob_list[0])
    for w, p in zip(weights, prob_list):
        blended_probs += w * p

    final_preds = blended_probs.argmax(axis=1)
    final_acc = accuracy_score(labels, final_preds)
    final_macro_f1 = f1_score(labels, final_preds, average="macro")

    print("\n" + "=" * 65)
    print("🏆 FINAL BLENDED ENSEMBLE RESULTS ON CMTE-NITANP 🏆")
    print(f"  Blended Accuracy : {final_acc * 100:.2f}%")
    print(f"  Blended Macro-F1 : {final_macro_f1 * 100:.2f}%")
    print(f"  Rayala & Seshadri Baseline (Electra) : 86.22% Acc | 88.69% F1")
    print("=" * 65)

    print("\nClassification Report:")
    print(classification_report(labels, final_preds, target_names=["Negative", "Neutral", "Positive"], digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, final_preds))


if __name__ == "__main__":
    main()
