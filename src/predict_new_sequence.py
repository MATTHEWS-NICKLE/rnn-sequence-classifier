import torch
import matplotlib.pyplot as plt

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

    return label, probability


def plot_prediction(
    sequence,
    label,
    probability,
    filename
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(sequence) + 1),
        sequence,
        marker="o"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Sequence Value")

    plt.title(
        f"RNN Prediction: {label}\n"
        f"Increasing Probability: {probability:.4f}"
    )

    plt.grid(True)

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


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

    label, probability = predict_sequence(
        new_sequence,
        "results/rnn_model_length10.pth"
    )

    print("Input sequence:")
    print(new_sequence)

    print(
        f"\nPredicted class: {label}"
    )

    print(
        f"Increasing probability: "
        f"{probability:.4f}"
    )

    plot_prediction(
        new_sequence,
        label,
        probability,
        "results/predictions.png"
    )

    print(
        "\nPrediction graph saved to "
        "results/predictions.png"
    )