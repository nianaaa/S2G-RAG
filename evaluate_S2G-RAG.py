import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score

from utils.evaluation import get_f1_score


def get_em(df):
    """Compute final-turn EM and F1 from the last row of each trajectory."""
    last_entries = df.groupby("ID").tail(1)
    em = accuracy_score(last_entries["Correct Answer"], [1] * len(last_entries)) * 100
    f1 = get_f1_score(last_entries) * 100

    print(f"S2G-RAG EM: {em:.3f}")
    print(f"S2G-RAG F1: {f1:.3f}")
    print(f"Total Data Points: {len(last_entries)}")
    return em


def evaluate_model(file_path):
    """Load predictions and report final-turn metrics."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    df_full = pd.read_csv(file_path)
    keep_ids = pd.unique(df_full["ID"])[:1000]
    df = df_full[df_full["ID"].isin(keep_ids)]
    get_em(df)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the model's EM and F1 based on predictions."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment name, same as name input to running the system",
    )

    args = parser.parse_args()
    file_path = f"predictions/{args.experiment_name}_predictions.csv"
    evaluate_model(file_path)


if __name__ == "__main__":
    main()
