import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split


def load_data(file_path, sequence_length):

    df = pd.read_csv(file_path)

    df = df[df["sequence_length"] == sequence_length]

    sequences = []
    labels = []

    for _, row in df.iterrows():

        sequence = [
            float(value)
            for value in row["sequence"].split(",")
        ]

        sequences.append(sequence)
        labels.append(int(row["label"]))

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train = torch.tensor(X_train).unsqueeze(-1)
    X_test = torch.tensor(X_test).unsqueeze(-1)

    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data(
        "dataset/sequence_data.csv",
        10
    )

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)