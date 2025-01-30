"""src: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, Normalize, ToTensor
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

INPUT_DIM = 39
BOTTLENECK_DIM = 4

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 16),
            nn.ELU(True),

            nn.Linear(16, 8),
            nn.ELU(True),

            nn.Linear(8, BOTTLENECK_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, 8),
            nn.ELU(True),

            nn.Linear(8, 16),
            nn.ELU(True),

            nn.Linear(16, INPUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once

    # configure our pipeline
    pipeline = Pipeline([('normalizer', Normalizer()),
                        ('scaler', MinMaxScaler())])
    #file = sorted(os.listdir("data"))[partition_id - 1]
    
    processed = pd.read_csv("processed.csv")

    partition_len = int(len(processed) / num_partitions)

    partition = processed.iloc[partition_id * partition_len: (partition_id + 1) * partition_len]

    clean = partition[partition[" Label"] == "BENIGN"]
    fraud = partition[partition[" Label"] != "BENIGN"]
    # Divide data on each node: 80% train, 20% test
    TRAINING_SAMPLE = int(0.85 * clean.shape[0])
    clean = clean.sample(frac=1).reset_index(drop=True)
    # training set: exlusively non-fraud transactions
    X_train = clean.iloc[:TRAINING_SAMPLE].drop(' Label', axis=1)
    # testing  set: the remaining non-fraud + all the fraud 
    #X_test = clean.iloc[TRAINING_SAMPLE:]
    #print(X_train.shape,X_test.shape)

    #X_test = pd.concat([X_test, fraud])
    X_train, X_validate = train_test_split(X_train, 
                                       test_size=0.05, train_size=0.95,
                                       random_state=1)
    pipeline.fit(X_train);
    X_train = pipeline.transform(X_train)
    X_validate = pipeline.transform(X_validate)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_validate = torch.tensor(X_validate, dtype=torch.float32)

    trainloader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(X_validate, batch_size=batch_size, drop_last=True)

    return trainloader, testloader


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            data = batch.to(device)
            #print("******* training ******** ",data.shape)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, data)
            loss.backward()
            #print("******* training loss ******** ",loss)
            optimizer.step()


    val_loss= test(net, valloader, device)

    results = {
        "val_loss": val_loss,
    }
    return results


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            inputs = batch.to(device)
            #print("******** inputs ********* ",inputs.shape)

            outputs = net(inputs)
            loss = criterion(outputs, inputs)  # Calculate loss for this batch
            total_loss += loss.item()  # Accumulate the total loss
            #print("******** outputs ********* ",outputs.shape)
            #print("******** loss ********* ",loss)
    avg_loss = total_loss / len(testloader)  # Calculate average loss
    print(f"Average MSE Loss: {avg_loss}")
    return avg_loss

