import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from prepare_data import load_data


class RNNClassifier(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1
    ):

        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        output, hidden = self.rnn(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction.squeeze(1)


def train_model(sequence_length):

    X_train, X_test, y_train, y_test = load_data(
        "dataset/sequence_data.csv",
        sequence_length
    )

    model = RNNClassifier()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    losses = []

    for epoch in range(100):

        model.train()

        optimizer.zero_grad()

        predictions = model(X_train)

        loss = criterion(predictions, y_train)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/100 "
                f"Loss: {loss.item():.4f}"
            )

    return model, losses, X_test, y_test


if __name__ == "__main__":

    model, losses, X_test, y_test = train_model(10)

    torch.save(
        model.state_dict(),
        "results/rnn_model_length10.pth"
    )

    plt.plot(losses)

    plt.title("RNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(
        "results/training_curve.png"
    )

    plt.close()

    print("Training completed.")