# mnist_experiments.py
#
# Sets up some classes and methods for performing experiments on the
# MNIST dataset.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time

# Just fixing some stupid loading issues
from importlib import reload
import beyond_gd as BGD
reload(BGD)

class MnistMLP(nn.Module):
    """Constructs a 3 layer perceptron network with skip connections.

    MnistMLP implements a multi-layer perceptron (MLP) with three
    layers, defaulting to taking in the 28 x 28 images that MNIST
    provides.

    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(28 * 28, 512)
        self.flatten = nn.Flatten()
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x1 = nn.ReLU()(self.lin1(x))
        x2 = nn.ReLU()(self.lin2(x1))
        x3 = self.lin3(x2)
        return x3

class SimpleConvNet(nn.Module):
    """Constructs a simple convolutional network for multiclass classification

    SimpleConvNet implements a convolutional NN for prediction
    (typically on MNIST) problems. This implements 3 convolutional
    layers followed by 2 fully connected layers.

    """
    def __init__(self, num_classes = 10):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   
        # Pooling layer used multiple times
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        # Third conv block
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def FitNN(model_type = "mlp",
          optimizer_type = "Adam",
          learning_rate = 1e-3):
    """Impelements Neural network fitting

    Arguments:
        model_type: The type of neural network to fit (one of MLP or Conv)
             MLP corresponds to the 3 layer perceptron network MnistMLP,
             while Conv corresponds to the SimpleConvNet class.
        optimizer_type: The type of optimizer to use. There are 5 optimizers
             to consider for this method:
             + SGD: Uses stochastic gradient method
             + Adam: Uses Adam
             + Adagrad: Uses AdaGrad
             + TruncatedSGD: Uses the truncated stochastic gradient method
             + TruncatedAda: Uses a truncated Adagrad method

             The two truncated methods, instead of simply using
             gradients, correct their stepsizes to reflect that the
             losses are nonnegative.
        learning_rate: the initial learning rate for the methods

    """
    # Figure out what devices are available, and use the one that's available.
    d_string = ("mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu")
    print("Setting backend device to use " + d_string)
    device = torch.device(d_string)
    # Load the training data
    train_dataset = datasets.MNIST(root="./data", train = True,
                                   download = True,
                                   transform = transforms.transforms.ToTensor())
    test_dataset = datasets.MNIST(root="./data", train = False,
                                  download = True,
                                  transform = transforms.transforms.ToTensor())
    
    # Choose model architecture based on input parameter
    if model_type.lower() == "mlp":
        model = MnistMLP().to(device)
    elif model_type.lower() == "conv" or model_type.lower() == "convnet":
        model = SimpleConvNet().to(device)
    else:
        raise ValueError("model_type must be either 'mlp' or 'conv'/'convnet'")

    # Always use multiclass logistic regression loss
    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == "adam":
        print("Using standard Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer_type.lower() == "sgd":
        print("Using standard SGD optimizer")
        optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    elif optimizer_type.lower() == "adagrad":
        print("Using standard Adagrad optimizer")
        optimizer = optim.Adagrad(model.parameters(), lr = learning_rate)
    elif (optimizer_type.lower() == "truncated" or
          optimizer_type.lower() == "truncatedsgd"):
        print("Using Truncated SGD optimizer")
        optimizer = BGD.TruncatedSGD(model.parameters(),
                                     lr = learning_rate)
    elif (optimizer_type.lower() == "truncatedada" or
          optimizer_type.lower() == "truncatedadagrad"):
        print("Using Truncated AdaGrad optimizer")
        optimizer = BGD.TruncatedAdagrad(model.parameters(),
                                         lr = learning_rate)        
    else:
        raise ValueError("optimizer_type must be one of 'adam'; 'adagrad';" +
                         " 'truncated' or 'truncatedSGD'; or " +
                         "'truncatedAda' or 'truncatedAdagrad'")

    final_loss = TrainLoop(train_dataset, test_dataset, model, criterion, optimizer, device)
    return final_loss

def TrainLoop(train_data, test_data, model, criterion, optimizer, device,
              num_epochs = 5,
              batch_size = 64):
    """Performs a numer of optimizer passes through the dataset

    This performs a certain number of passes (num_epochs) through the
    entire dataset given in train_data. At the end of each epoch/pass,
    evaluates the test losses on the data in test_data. Also prints the
    time it takes to run each epoch.

    """
    train_loader = DataLoader(train_data,
                              batch_size = batch_size, shuffle=True)
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Zero out the total losses and start training
        model.train()
        total_loss = 0
        for (inputs, labels) in train_loader:
            # Send the data to the actual device (i.e., GPU):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Define a closure that computes the loss to pass in to
            # the optimizer methods. This is useful for things like
            # truncated methods (or more advanced optimization
            # methods)
            def loss_closure():
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and then do an optimizer step
                loss.backward()
                return loss

            # Now take optimization step
            loss = optimizer.step(loss_closure)

            total_loss += loss.item()

        epoch_end = time.time()
        print(f"Epoch [{epoch + 1} / {num_epochs}]. " +
              f"Mean loss: {total_loss / len(train_loader):.4f}")
        print(f"Time for epoch {epoch + 1}: " +
              f"{epoch_end - epoch_start:.2f} seconds")
        
        # Evaluate on test set
        accuracy = TestEval(test_data, model, device)
        print(f"Test accuracy: {accuracy:.2f}%\n")
    return total_loss

def TestEval(test_data, model, device):
    model.eval()
    correct = 0
    total = 0
    test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total



if __name__ == "__main__":
    # FitMLP("mlp")  # Can now specify "mlp" or "conv"/"convnet"
    FitNN("conv")
