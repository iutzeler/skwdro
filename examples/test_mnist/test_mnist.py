import tqdm
import numpy as np
import torch as pt
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from model import make_alexnet

root = "examples/test_mnist/data/"
device = "cuda" if pt.cuda.is_available() else "cpu"
pt.set_float32_matmul_precision('high')


def accuracy(output, target):
    return (
        output
        .detach()
        .cpu()
        .argmax(dim=1)
        .eq(target.cpu())
        .float()
        .mean()
        .item()*100
    )


def step(model, features, target, criterion, optimizer):
    features = features.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    classes = model(features)

    loss = criterion(classes, target).mean()
    loss.backward()

    optimizer.step()
    return loss.detach().item()


@pt.no_grad()
def evalnet(model, features, target, criterion):
    features = features.to(device)
    target = target.to(device)

    model.eval()
    classes = model(features)
    model.train()
    return (
        accuracy(classes, target),
        criterion(classes, target).mean().detach().item()
    )


def traineval_loop(model, train_loader, test_loader, criterion, optimizer):
    # === Train ===
    nested_it = tqdm.tqdm(train_loader, position=1, leave=False)
    for features, target in nested_it:
        loss = step(model, features, target, criterion, optimizer)
        nested_it.set_postfix({"trl": f"{loss:.2e}"})

    # === Eval ===
    nested_it = tqdm.tqdm(test_loader, position=1, leave=False)
    acc, ls = list(zip(*[
        evalnet(
            model, features, target, criterion
        ) for features, target in nested_it
    ]))
    return np.mean(acc), np.mean(ls)


def train_alexnet(model, dataset_train, dataset_test, n_epochs: int = 100):
    optimizer = pt.optim.Adam(model.parameters(), lr=1e-2)
    criterion = pt.nn.CrossEntropyLoss(reduction='none')

    bs = 256
    train_loader = DataLoader(
        dataset_train,
        batch_size=bs,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=bs,
        shuffle=True,
    )

    it = tqdm.tqdm(range(n_epochs), position=0)
    mean_losses = []
    max_acc = 0.
    for epoch in it:
        acc, avgloss = traineval_loop(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer
        )
        if acc > max_acc:
            max_acc = acc
            pt.save(model.state_dict(), root+"weights.pt")
        it.set_postfix({"xH": f"{avgloss:.2f}", "acc": f"{acc:.2f}%"})
        mean_losses.append(avgloss)
    return mean_losses


def main():
    model = make_alexnet(device).to(device)
    model.load_state_dict(pt.load(root+"weights.pt"), strict=False)

    dataset_train = MNIST(
        root, download=True, train=True, transform=model.preprocess
    )
    dataset_test = Subset(
        MNIST(root, download=True, train=False, transform=model.preprocess),
        range(0, 10000, 100)
    )

    np.save(
        root+"losses.npy",
        train_alexnet(pt.compile(model), dataset_train, dataset_test, 10)
    )


if __name__ == '__main__':
    main()
