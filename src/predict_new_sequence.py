import torch

from train_rnn import RNNClassifier


def predict_sequence(sequence, model_path):

    model = RNNClassifier()

    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True
        )
    )

    model.eval()

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).view(1, len(sequence), 1)

    with torch.no_grad():

        output = model(x)

        probability = torch.sigmoid(output).item()

    if probability >= 0.5:
        label = "Increasing"
    else:
        label = "Decreasing"

    print("Input sequence:")
    print(sequence)

    print(
        f"Predicted class: {label}"
    )

    print(
        f"Increasing probability: "
        f"{probability:.4f}"
    )


if __name__ == "__main__":

    new_sequence = [
        2.0,
        3.1,
        4.0,
        5.2,
        6.1,
        7.0,
        8.2,
        9.1,
        10.0,
        11.2
    ]

    predict_sequence(
        new_sequence,
        "results/rnn_model_length10.pth"
    )