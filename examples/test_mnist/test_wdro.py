import tqdm
import numpy as np
import torch as pt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision.datasets import MNIST

from model import make_alexnet

from skwdro.wrap_problem import dualize_primal_loss

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
    oh_target = F.one_hot(target).to(features)

    optimizer.zero_grad()

    # classes = model(features)

    # loss = criterion(classes, oh_target).mean()
    loss = model(features, oh_target, reset_sampler=True).mean()
    loss.backward()

    optimizer.step()
    return loss.detach().item()


@pt.no_grad()
def evalnet(model, features, target, criterion):
    features = features.to(device)
    target = target.to(device)

    model.eval()
    classes = model.primal_loss.transform(features)
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


def get_warmup_batch(ds: MNIST, size: int, rng: pt.Generator):
    batch_ = default_collate([ds[i] for i in pt.randint(
        len(ds), (size,), generator=rng, device=rng.device
    )])
    return (
        batch_[0].to(device),
        F.one_hot(batch_[1]).to(device, batch_[0].dtype)
    )


def train_alexnet(model, dataset_train, dataset_test, n_epochs: int = 100):
    rng = pt.Generator(device='cpu')

    optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)
    criterion = pt.nn.CrossEntropyLoss(reduction='none')

    bs = 128
    train_loader = DataLoader(
        dataset_train,
        batch_size=bs,
        shuffle=True,
        generator=rng,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=bs,
        shuffle=True,
        generator=rng,
    )

    warmup = get_warmup_batch(dataset_train, bs, rng)
    assert len(warmup) == 2
    wdro_model = dualize_primal_loss(
        criterion,
        model,
        pt.tensor(1e-5).to(device),
        warmup[0],
        warmup[1],

        cost_spec="t-NC-2-2",
        sigma=1e-8
    )

    it = tqdm.tqdm(range(n_epochs), position=0)
    mean_losses = []
    max_acc = 0.
    for epoch in it:
        wdro_model.get_initial_guess_at_dual(*warmup)
        acc, avgloss = traineval_loop(
            wdro_model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
        )

        if acc > max_acc:
            max_acc = acc
            pt.save(wdro_model.state_dict(), root+"_dro_weights.pt")
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
        root+"losses_dro.npy",
        train_alexnet(model, dataset_train, dataset_test, 100)
    )


if __name__ == '__main__':
    main()
