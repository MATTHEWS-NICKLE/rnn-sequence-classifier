import torch

from train_rnn import RNNClassifier
from prepare_data import load_data


def evaluate(sequence_length):

    X_train, X_test, y_train, y_test = load_data(
        "dataset/sequence_data.csv",
        sequence_length
    )

    model = RNNClassifier()

    model.load_state_dict(
        torch.load(
            f"results/rnn_model_length{sequence_length}.pth",
            weights_only=True
        )
    )

    model.eval()

    with torch.no_grad():

        outputs = model(X_test)

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        accuracy = (
            predictions == y_test
        ).float().mean()

    print(
        f"Sequence length {sequence_length}: "
        f"{accuracy.item() * 100:.2f}% accuracy"
    )


if __name__ == "__main__":

    evaluate(10)