r"""
Simple Neural Network
=====================

This example solves a simple binary classification problem using a basic neural network with 2 layers. 


# .. math::

#     f^\star(x) = \frac{10}{e^{x}+e^{-x}} + x

# from :math:`n=100` samples uniformly drawn from :math:`[-2,2]` and corrupted by a Gaussian noise with zero mean and variance :math:`0.1`. 


"""
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import plot_decision_boundary
import torch as pt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from skwdro.wrap_problem import dualize_primal_loss
from skwdro.solvers.oracle_torch import DualLoss
from skwdro.base.losses_torch.wrapper import WrappedPrimalLoss

# %%
# Problem setup
# ~~~~~~~~~~~~~

from sklearn.datasets import make_moons

n = 200

X, y = make_moons(n_samples=n,
                  noise=0.1,
                  random_state=42)

# Visualize the data
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu);
#plt.show()

# Turn data into tensors
X = pt.tensor(X, dtype=pt.float)
y = pt.tensor(y, dtype=pt.float)

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)


dataset = DataLoader(TensorDataset(pt.tensor(X_train), pt.Tensor(y_train)))

device = "cuda" if pt.cuda.is_available() else "cpu"

# %%
# Polynomial model
# ~~~~~~~~~~~~~~~~

class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_features),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


    # def __init__(self, in_features, out_features, hidden_units):
    #     super().__init__()
        
    #     self.layer1 = nn.Linear(in_features=in_features, 
    #                              out_features=hidden_units)
    #     self.layer2 = nn.Linear(in_features=hidden_units, 
    #                              out_features=hidden_units)
    #     self.layer3 = nn.Linear(in_features=hidden_units,
    #                             out_features=out_features)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

model = SimpleNN(in_features=2,
                      out_features=1,
                      hidden_units=5).to(device)


print(model)


loss_fn = nn.BCEWithLogitsLoss()

# Define a sample batch for initialization
sample_batch_x, sample_batch_y = next(iter(dataset))

# Robust loss
robust_loss = dualize_primal_loss( 
            loss_fn,
            model,
            pt.tensor(1.0),
            sample_batch_x, sample_batch_y
        ) # Replaces the loss of the model by the dual WDRO loss

# %%
# Training loop
# ~~~~~~~~~~~~~
pt.manual_seed(42)
epochs=100


optimizer = pt.optim.AdamW(params=robust_loss.parameters(),lr=1e-2)


# Training loop
for epoch in range(epochs):
    for batch_x, batch_y in dataset:

        ### Training
        model.train()

        optimizer.zero_grad()
        loss = robust_loss(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

        ### Testing
        model.eval() 
        with pt.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test).squeeze()
            test_pred = pt.round(pt.sigmoid(test_logits))
            # 2. Caculate the loss/acc
            test_loss = loss_fn(test_logits, y_test)

        # Print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.2f} | Test loss: {test_loss:.2f} ")


# # Loop through the data
# for epoch in range(epochs):
#   ### Training
#   model.train()

#   # 1. Forward pass
#   y_logits = model(X_train).squeeze()
#   y_pred_probs = pt.sigmoid(y_logits)
#   y_pred = pt.round(y_pred_probs)

#   # 2. Calculate the loss
#   loss = loss_fn(y_logits, y_train) # loss = compare model raw outputs to desired model outputs

#   # 3. Zero the gradients
#   optimizer.zero_grad()

#   # 4. Loss backward (perform backpropagation) 
#   loss.backward()

#   # 5. Step the optimizer (gradient descent)
#   optimizer.step()

#   ### Testing
#   model.eval() 
#   with pt.inference_mode():
#     # 1. Forward pass
#     test_logits = model(X_test).squeeze()
#     test_pred = pt.round(pt.sigmoid(test_logits))
#     # 2. Caculate the loss/acc
#     test_loss = loss_fn(test_logits, y_test)

#   # Print out what's happening
#   if epoch % 10 == 0:
#     print(f"Epoch: {epoch} | Loss: {loss:.2f} | Test loss: {test_loss:.2f} ")


# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()