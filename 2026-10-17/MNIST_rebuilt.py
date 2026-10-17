from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parent
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42


def load_mnist(flatten=True, normalize=True):

    train = MNIST(root=str(ROOT / "data"), train=True, download=True)
    test = MNIST(root=str(ROOT / "data"), train=False, download=True)

    def convert(dataset):
        images = dataset.data.numpy()
        labels = dataset.targets.numpy()
        if flatten:
            images = images.reshape(-1, 784)
        else:
            images = images.reshape(-1, 1, 28, 28)
        if normalize:
            images = images.astype(np.float32) / 255.0
        return images, labels

    return convert(train), convert(test)


def create_network():

    return nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)


    x_train = torch.from_numpy(x_train.astype(np.float32) / 255.0)
    x_test = torch.from_numpy(x_test.astype(np.float32) / 255.0)
    t_train = torch.from_numpy(t_train).long()
    t_test = torch.from_numpy(t_test).long()
    train_loader = DataLoader(
        TensorDataset(x_train, t_train), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(TensorDataset(x_test, t_test), batch_size=512)


    network = create_network().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    print(f"Device: {device}", flush=True)

    for epoch in range(EPOCHS):
        network.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = network(images)
            loss = loss_function(scores, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        print(f"Epoch {epoch + 1}/{EPOCHS}, loss={loss_sum / len(x_train):.4f}", flush=True)


    network.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = network(images.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
        predictions = network(x_test[:10].to(device)).argmax(dim=1).cpu()

    accuracy = correct / len(t_test)
    print(f"Test accuracy: {accuracy:.2%} ({correct}/{len(t_test)})")
    print("First 10 predictions:", predictions.tolist())
    print("First 10 labels:     ", t_test[:10].tolist())


    model_path = ROOT / "mnist_rebuilt_model.pt"
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in network.state_dict().items()},
        "architecture": [784, 128, 64, 10],
        "normalization": "pixel / 255",
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "test_accuracy": accuracy,
    }, model_path)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
