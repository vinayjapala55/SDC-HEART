"""
Patient-level Stratified Splitting for the BMD-HS Heart Sound Dataset.

Features:
- Enforces multi-label stratified partition using patient-level label combinations (strata).
- Guarantees strict patient isolation across splits (Train: 76, Val: 17, Test: 16 = 109 total).
- Pins training-augmented patients and patient_022 strictly to the training split.
- Fairly distributes Normal (N) and all valvular disease classes across Train, Val, and Test.
- Preserves patient_022 unannotated/MVP status as -1 (unknown target) rather than false negative.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = ROOT_DIR / "train.csv"
OUT_DIR = ROOT_DIR / "data" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["AS", "AR", "MR", "MS", "N"]

def generate_patient_splits(random_state: int = 42):
    df = pd.read_csv(SOURCE_CSV)
    
    # Identify augmented patients from manifest if present
    aug_manifest_path = ROOT_DIR / "dataset" / "metadata" / "augmentation_manifest.csv"
    if aug_manifest_path.exists():
        aug_df = pd.read_csv(aug_manifest_path)
        aug_pids = set(aug_df["patient_id"].unique())
    else:
        aug_pids = set()

    # Pin augmented patients and patient_022 to train to prevent leakage
    pinned_train = aug_pids | {"patient_022"}

    unpinned = df[~df["patient_id"].isin(pinned_train)].copy().reset_index(drop=True)
    unpinned["stratum"] = unpinned[LABEL_COLS].astype(str).agg(lambda r: "|".join(r), axis=1)

    # 108 patients total excluding patient_022: target train=75, val=17, test=16
    ratios = {"train": (76 - len(pinned_train)) / len(unpinned), "val": 17 / len(unpinned), "test": 16 / len(unpinned)}

    np.random.seed(random_state)
    splits = {"train": list(pinned_train), "val": [], "test": []}

    for stratum, group in unpinned.groupby("stratum", sort=False):
        n = len(group)
        shuffled = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        pids = shuffled["patient_id"].tolist()
        
        n_val = int(round(n * ratios["val"]))
        n_test = int(round(n * ratios["test"]))
        n_train = n - n_val - n_test
        if n_val + n_test > n:
            n_val = min(n_val, n)
            n_test = n - n_val
            n_train = 0
        splits["val"].extend(pids[:n_val])
        splits["test"].extend(pids[n_val:n_val+n_test])
        splits["train"].extend(pids[n_val+n_test:])

    # Exact boundary adjustments
    while len(splits["val"]) > 17:
        splits["train"].append(splits["val"].pop())
    while len(splits["test"]) > 16:
        splits["train"].append(splits["test"].pop())
    while len(splits["test"]) < 16:
        for p in list(splits["train"]):
            if p not in pinned_train and p not in splits["val"] and p not in splits["test"]:
                splits["train"].remove(p)
                splits["test"].append(p)
                break
    while len(splits["val"]) < 17:
        for p in list(splits["train"]):
            if p not in pinned_train and p not in splits["val"] and p not in splits["test"]:
                splits["train"].remove(p)
                splits["val"].append(p)
                break

    splits["train"].sort()
    splits["val"].sort()
    splits["test"].sort()

    def build_split_df(pids):
        sub = df[df["patient_id"].isin(pids)][["patient_id", *LABEL_COLS]].copy().sort_values("patient_id").reset_index(drop=True)
        for col in LABEL_COLS:
            sub[col] = sub[col].fillna(-1).astype(int)
        return sub

    train_df = build_split_df(splits["train"])
    val_df = build_split_df(splits["val"])
    test_df = build_split_df(splits["test"])

    train_df.to_csv(OUT_DIR / "train_patients.csv", index=False)
    val_df.to_csv(OUT_DIR / "val_patients.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_patients.csv", index=False)

    return train_df, val_df, test_df

if __name__ == "__main__":
    tr, va, te = generate_patient_splits()
    print(f"Generated patient splits: Train={len(tr)}, Val={len(va)}, Test={len(te)}")
