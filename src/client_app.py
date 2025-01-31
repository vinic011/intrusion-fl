"""src: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.task import Net, get_weights, load_data, set_weights, test, train, eval


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, testloader, local_epochs, learning_rate, y_test):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        thresholds = [4.0, 4.2, 3.8]
        loss = eval(self.net, self.testloader, self.y_test, self.device, thresholds)
        return loss, len(self.valloader.dataset), {"loss": loss}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader, testloader, y_test= load_data(partition_id, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, testloader, local_epochs, learning_rate, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
