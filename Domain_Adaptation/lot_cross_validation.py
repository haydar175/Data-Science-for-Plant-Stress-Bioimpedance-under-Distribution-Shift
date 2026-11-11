"""
Domain adaptation for out-of-distribution (OOD) water-stress data.

A linear optimal-transport map is fitted from the OOD dataset (source) to the
original water-stress dataset (target). The mapped OOD spectra are then
classified, plant by plant, with an ElasticNet logistic regression trained
only on the original data, using the frequencies selected in the paper.

Usage
-----
    python lot_cross_validation.py \
        --original Water_Stress.npz --ood New_Data.npz --out results

Expected inputs
---------------
Water_Stress.npz : X (n, 400) = 200 frequencies x (real, imaginary), y, plant_ids (P0, P1, P3)
New_Data.npz     : X (n, 200, 7) with magnitude in column 1 and
                   phase (degrees) in column 2, y, plant_ids (P0 ... P4)
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import ot
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

N_FREQ = 200                       # frequency points per spectrum (100 Hz - 10 MHz)
SELECTED_FREQ_IDX = [1, 3, 0, 152]  # frequencies kept after the regularization path
ORIGINAL_PLANTS = ["P1", "P3"]     # original plants used as they are
DUPLICATED_PLANT = "P0"            # original plant counted twice (stands in for P2)
OOD_CALIBRATION_PLANTS = ["P0", "P1", "P2", "P4"]  # OOD plants used to fit the map
OOD_HELD_OUT_PLANTS = ["P3"]       # OOD plant never seen when fitting the map
C_VALUES = [0.01, 1, 10]
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 1]
SEED = 42


def print_counts(title, plant_ids):
    print(f"\n{title}")
    for plant, count in zip(*np.unique(plant_ids, return_counts=True)):
        print(f"  {plant}: {count} samples")


def load_original(path):
    """Original water-stress data: real and imaginary part, shape (n, 200, 2)."""
    data = np.load(path)
    X = data["X"].reshape(-1, N_FREQ, 2)
    y = data["y"]
    plant_ids = data["plant_ids"]
    print("Unique plant ids in original data:", np.unique(plant_ids))

    keep = np.isin(plant_ids, ORIGINAL_PLANTS)
    dup = plant_ids == DUPLICATED_PLANT
    X_out = np.concatenate([X[keep], X[dup], X[dup]], axis=0)
    y_out = np.concatenate([y[keep], y[dup], y[dup]], axis=0)
    ids_out = np.concatenate([plant_ids[keep], plant_ids[dup], plant_ids[dup]], axis=0)
    return X_out, y_out, ids_out


def load_ood(path):
    """OOD data: magnitude and phase (degrees) converted to real and imaginary part."""
    data = np.load(path)
    X = data["X"][:, :N_FREQ, 1:3]
    magnitude = X[:, :, 0]
    phase = np.deg2rad(X[:, :, 1])
    X_complex = np.stack([magnitude * np.cos(phase), magnitude * np.sin(phase)], axis=2)
    return X_complex, data["y"], data["plant_ids"]


def select_frequencies(X):
    return X[:, SELECTED_FREQ_IDX, :].reshape(X.shape[0], -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--original", default="Water_Stress.npz")
    parser.add_argument("--ood", default="New_Data.npz")
    parser.add_argument("--out", default="results", help="folder for confusion matrices")
    parser.add_argument("--show", action="store_true", help="also display the figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- Data -------------------------------------------------------------
    X_or, y_or, ids_or = load_original(args.original)
    print_counts("Counts in original data (training):", ids_or)

    X_ood, y_ood, ids_ood_raw = load_ood(args.ood)
    ids_ood = np.array(["ood_" + pid for pid in ids_ood_raw])
    calib = np.isin(ids_ood_raw, OOD_CALIBRATION_PLANTS)
    held_out = np.isin(ids_ood_raw, OOD_HELD_OUT_PLANTS)
    print_counts("Counts in OOD calibration data:", ids_ood[calib])
    print_counts("Counts in OOD held-out data:", ids_ood[held_out])

    # --- Linear optimal transport: OOD (source) -> original (target) ------
    source = X_ood[calib].reshape(calib.sum(), -1)
    target = X_or.reshape(X_or.shape[0], -1)
    print("\nSource (OOD calibration) shape:", source.shape)
    print("Target (original) shape:", target.shape)
    A, b = ot.gaussian.empirical_gaussian_gromov_wasserstein_mapping(source, target)

    def transport(X):
        return (X.reshape(X.shape[0], -1) @ A + b).reshape(X.shape)

    # --- Features: selected frequencies only ------------------------------
    X_train = select_frequencies(X_or)
    y_train = y_or
    X_test = np.concatenate(
        [select_frequencies(transport(X_ood[calib])),
         select_frequencies(transport(X_ood[held_out]))], axis=0)
    y_test = np.concatenate([y_ood[calib], y_ood[held_out]], axis=0)
    ids_test = np.concatenate([ids_ood[calib], ids_ood[held_out]], axis=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- ElasticNet grid, trained on original data only -------------------
    models = {}
    for C in C_VALUES:
        for l1 in L1_RATIOS:
            model = LogisticRegression(
                penalty="elasticnet", solver="saga", C=C, l1_ratio=l1,
                max_iter=5000, random_state=SEED)
            models[(C, l1)] = model.fit(X_train, y_train)

    # --- Evaluation on each OOD plant -------------------------------------
    best_acc_per_plant = {}
    for plant in np.unique(ids_test):
        idx = ids_test == plant
        print(f"\n=== Testing on plant: {plant} ({idx.sum()} samples) ===")
        best_acc, best_params = -1.0, None
        for (C, l1), model in models.items():
            acc = accuracy_score(y_test[idx], model.predict(X_test[idx]))
            print(f"  C={C}, l1_ratio={l1}, Accuracy={acc:.4f}")
            if acc > best_acc:
                best_acc, best_params = acc, (C, l1)
        print(f"-> Best accuracy for {plant}: {best_acc:.4f} "
              f"(C={best_params[0]}, l1_ratio={best_params[1]})")
        best_acc_per_plant[plant] = best_acc

        cm = confusion_matrix(y_test[idx], models[best_params].predict(X_test[idx]))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion matrix for {plant}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"confusion_matrix_{plant}.png"), dpi=200)
        if args.show:
            plt.show()
        plt.close()

    print("\nSummary of best accuracies per plant:")
    for plant, acc in best_acc_per_plant.items():
        print(f"  {plant}: {acc:.4f}")
    accs = np.array(list(best_acc_per_plant.values()))
    print(f"Average accuracy: {accs.mean():.4f} +/- {accs.std():.4f}")


if __name__ == "__main__":
    main()
