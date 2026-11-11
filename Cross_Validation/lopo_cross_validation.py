"""
Leave-one-plant-out cross-validation from the .npz files.

Each plant is held out in turn. An ElasticNet logistic regression is trained
on the remaining plants, using only the frequencies selected in the paper;
hyperparameters are chosen on a validation split of the training plants.

Usage
-----
    python lopo_cross_validation.py --data Water_Stress.npz --stress water
    python lopo_cross_validation.py --data Iron_Stress.npz --stress iron

Expected input
--------------
X (n, 400) = 200 frequencies x (real, imaginary), y (0 control, 1 early,
2 late), plant_ids.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import MinMaxScaler

N_FREQ = 200
SELECTED_FREQ_IDX = {
    "water": [1, 0, 3],        # 100 - 120 Hz
    "iron": [198, 199, 187],   # 4.9 - 10 MHz
}
C_VALUES = [0.1, 1, 5, 10, 20, 40, 50, 100]
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 1]
CLASS_NAMES = ["Control", "Early", "Late"]
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="Water_Stress.npz or Iron_Stress.npz")
    parser.add_argument("--stress", required=True, choices=sorted(SELECTED_FREQ_IDX))
    parser.add_argument("--full-spectrum", action="store_true",
                        help="use all 200 frequencies instead of the selected ones (slow)")
    parser.add_argument("--out", default="results", help="folder for confusion matrices")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.data)
    X = data["X"].reshape(-1, N_FREQ, 2)
    y, plant_ids = data["y"], data["plant_ids"]
    idx = list(range(N_FREQ)) if args.full_spectrum else SELECTED_FREQ_IDX[args.stress]
    X = X[:, idx, :].reshape(X.shape[0], -1)
    print(f"Data: {X.shape}, plants: {np.unique(plant_ids)}")

    test_accuracies = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups=plant_ids):
        plant = plant_ids[test_idx][0]
        X_train, X_val, y_train, y_val = train_test_split(
            X[train_idx], y[train_idx], test_size=0.3,
            stratify=y[train_idx], random_state=SEED)
        scaler = MinMaxScaler().fit(X_train)
        X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X[test_idx]))

        best_val, best_model = -1.0, None
        for C in C_VALUES:
            for l1 in L1_RATIOS:
                model = LogisticRegression(
                    penalty="elasticnet", solver="saga", C=C, l1_ratio=l1,
                    max_iter=2000, random_state=SEED).fit(X_train, y_train)
                val_acc = accuracy_score(y_val, model.predict(X_val))
                if val_acc > best_val:
                    best_val, best_model = val_acc, model

        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y[test_idx], y_pred)
        test_accuracies.append(test_acc)
        print(f"Test plant {plant}: validation {best_val:.4f}, test {test_acc:.4f} "
              f"(C={best_model.C}, l1_ratio={best_model.l1_ratio})")

        cm = confusion_matrix(y[test_idx], y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f"{args.stress.capitalize()} stress, test plant {plant}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"confusion_matrix_{args.stress}_{plant}.png"), dpi=200)
        plt.close()

    print(f"Mean test accuracy: {np.mean(test_accuracies):.4f} +/- {np.std(test_accuracies):.4f}")


if __name__ == "__main__":
    main()
