import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from model import make_alexnet

from skwdro.wrap_problem import dualize_primal_loss

LR_MAX = 1e-2
MEAN_AN = np.array([0.485, 0.456, 0.406])
STD_AN = np.array([0.229, 0.224, 0.225])

def prepare(X, y, upscale=256):
    #st = StandardScaler(copy=False)
    st = MinMaxScaler(copy=False)
    end_wh = (upscale // 8) * 8
    _X_scaled = st.fit_transform(X)[:, None, :] * STD_AN[None, :, None] + MEAN_AN[None, :, None]
    _X_interpolation = nn.functional.interpolate(pt.from_numpy(_X_scaled.reshape(-1, 3, 8, 8)), size=(end_wh, end_wh), mode='nearest').numpy()
    X = _X_interpolation.reshape(X.shape[0], -1).clip(0., 1.)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.array(y, dtype=np.int64)

    X = X.astype(np.float32)
    return X, y

def experiment(n_epochs, n_experiments=1000):
    X, y = prepare(*load_digits(return_X_y=True)) # type: ignore
    train_losses, test_losses = list(zip(train_loop_alexnet(X, y, n_epochs) for _ in range(n_experiments)))
    plt.hist(train_losses, density=True, c='r')
    plt.hist(test_losses, density=True, c='g')

def train_dual_alexnet(rho: pt.Tensor, X, y, epochs=10, verbose=False, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # Loss is cross entropy (KLdiv without cst entropy of target)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Datasets
    train_set = TensorDataset(pt.from_numpy(X_train), pt.from_numpy(y_train))
    test_set = TensorDataset(pt.from_numpy(X_test), pt.from_numpy(y_test))

    # Model is our modified alexnet with dual loss
    primal_model = make_alexnet()

    # Dualize model
    starting_xi, starting_labels = train_set[np.random.randint(0, len(train_set))]
    model = dualize_primal_loss(
            criterion,
            primal_model,
            rho,
            starting_xi,
            starting_labels
        ).to(device)
    model.train()

    # Optimizer setup with adam
    optimizer = pt.optim.Adam(model.parameters(), lr=LR_MAX)

    track_train_losses = []
    track_test_losses = []
    track_test_accs = []

    # One epoch = on loop on test set
    for epoch in range(epochs):
        _eqline = "=" * (7 + len(str(epoch)+str(epochs)))
        if verbose:
            print(_eqline)
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print(_eqline)

        # Test batches loader
        test_loader = DataLoader(test_set, batch_size=128, shuffle=True)

        lr = LR_MAX / (1. + .1 * np.sqrt(epoch))
        optimizer.param_groups[0]['lr'] = lr

        # First epoch: stop erm
        if epoch == 0:
            model.erm_mode = True
        elif epoch == 1:
            model.eval()
            model.get_initial_guess_at_dual(starting_xi, starting_labels)
            model.erm_mode = False
            model.train()

        # Loop on test data
        for test_data, test_target in test_loader:
            # Train batches loader
            train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

            # Loop on train data once per test batch
            for data, target in train_loader:
                # Training step
                model.zero_grad()
                loss = model(data.to(device), target.to(device))
                loss.backward()
                optimizer.step()

                # Print metrics
                l = loss.cpu().item()
                if verbose: print("   Train loss: {:.2f}".format(l))
                track_train_losses.append(l)

            # Start evaluation on test data
            model.eval()
            output = primal_model(test_data.to(device))
            loss = criterion(output, test_target.to(device)).mean()

            # Print metrics
            l = loss.cpu().item()
            a = output.cpu().argmax(dim=1).eq(test_target).float().mean().item()*100
            if verbose: print("Test loss: {:.2f}".format(l), "acc: {:.2f}%".format(a))
            track_test_losses.append(l)
            track_test_accs.append(a)
            model.train()

    if plot:
        # Loss plot
        plt.subplot(2, 1, 1)
        plt.plot(track_train_losses)
        plt.plot(np.arange(1, 1 + len(track_test_losses)) * (len(track_train_losses) // len(track_test_losses)), track_test_losses)
        plt.yscale("log")

        # Accuracy plot
        plt.subplot(2, 1, 2)
        plt.plot(track_test_accs)

        plt.show()
    return np.mean(track_train_losses[-10:]), np.mean(track_test_losses[-10:])


def train_loop_alexnet(X, y, epochs=10, verbose=False, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # Model is our modified alexnet
    model = make_alexnet().to(device)
    model.train()

    # Loss is cross entropy (KLdiv without cst entropy of target)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Optimizer setup with adam
    optimizer = pt.optim.Adam(model.parameters(), lr=LR_MAX)

    # Datasets
    train_set = TensorDataset(pt.from_numpy(X_train), pt.from_numpy(y_train))
    test_set = TensorDataset(pt.from_numpy(X_test), pt.from_numpy(y_test))

    track_train_losses = []
    track_test_losses = []
    track_train_accs = []
    track_test_accs = []

    # One epoch = on loop on test set
    for epoch in range(epochs):
        _eqline = "=" * (7 + len(str(epoch)+str(epochs)))
        if verbose:
            print(_eqline)
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print(_eqline)

        # Test batches loader
        test_loader = DataLoader(test_set, batch_size=128, shuffle=True)

        lr = LR_MAX / (1. + .1 * np.sqrt(epoch))
        optimizer.param_groups[0]['lr'] = lr

        # Loop on test data
        for test_data, test_target in test_loader:
            # Train batches loader
            train_loader = DataLoader(train_set, batch_size=512, shuffle=True)

            # Loop on train data once per test batch
            for data, target in train_loader:
                # Training step
                model.zero_grad()
                output = model(data.to(device))
                loss = criterion(output, target.to(device)).mean()
                loss.backward()
                optimizer.step()

                # Print metrics
                l = loss.cpu().item()
                a = output.cpu().argmax(dim=1).eq(target).float().mean().item()*100
                if verbose: print("   Train loss: {:.2f}".format(l), "acc: {:.2f}%".format(a))
                track_train_losses.append(l)
                track_train_accs.append(a)

            # Start evaluation on test data
            model.eval()
            output = model(test_data.to(device))
            loss = criterion(output, test_target.to(device)).mean()

            # Print metrics
            l = loss.cpu().item()
            a = output.cpu().argmax(dim=1).eq(test_target).float().mean().item()*100
            if verbose: print("Test loss: {:.2f}".format(l), "acc: {:.2f}%".format(a))
            track_test_losses.append(l)
            track_test_accs.append(a)
            model.train()

    if plot:
        # Loss plot
        plt.subplot(2, 1, 1)
        plt.plot(track_train_losses)
        plt.plot(np.arange(1, 1 + len(track_test_losses)) * (len(track_train_losses) // len(track_test_losses)), track_test_losses)
        plt.yscale("log")

        # Accuracy plot
        plt.subplot(2, 1, 2)
        plt.plot(track_train_accs)
        plt.plot(np.arange(1, 1 + len(track_test_accs)) * (len(track_train_accs) // len(track_test_accs)), track_test_accs)

        plt.show()
    return np.mean(track_train_losses[-10:]), np.mean(track_test_losses[-10:])

if __name__ == "__main__":
    #experiment(20, 100)
    train_dual_alexnet(pt.tensor(1e-3), *prepare(*load_digits(return_X_y=True)), verbose=True) # type: ignore
