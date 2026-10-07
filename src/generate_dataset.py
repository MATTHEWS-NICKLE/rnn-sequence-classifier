import numpy as np
import pandas as pd


def generate_sequence(length, label):
    start = np.random.uniform(1, 10)

    if label == 1:
        # Increasing sequence
        step = np.random.uniform(0.5, 1.5)
    else:
        # Decreasing sequence
        step = -np.random.uniform(0.5, 1.5)

    sequence = []

    for i in range(length):
        value = start + (i * step)
        value += np.random.normal(0, 0.15)
        sequence.append(round(value, 3))

    return sequence


def create_dataset():
    np.random.seed(42)

    rows = []

    samples_per_class = 100

    for length in [5, 10, 20]:

        for label in [0, 1]:

            for sample in range(samples_per_class):

                sequence = generate_sequence(length, label)

                rows.append({
                    "sequence_length": length,
                    "sequence": ",".join(map(str, sequence)),
                    "label": label
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = create_dataset()

    df.to_csv(
        "dataset/sequence_data.csv",
        index=False
    )

    print("Dataset created successfully.")
    print("Total samples:", len(df))
    print(df.head())