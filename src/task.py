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
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


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

def select_cols():
    data_folder = "/Users/viniciuspereira/Documents/fl/instrusion-fl/data"
    all_files = os.listdir(data_folder)

    csv_files = [file for file in all_files if file.endswith('.csv')]

    dataframes = [pd.read_csv(os.path.join(data_folder, file)) for file in csv_files]

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Drop columns with only one unique value
    drop_cols = []
    for cols in combined_df.columns:
        if combined_df[cols].nunique() < 2:
            drop_cols.append(cols)
    combined_df.drop(drop_cols, axis=1, inplace=True)
    
    combined_all_num = combined_df.copy()
    combined_all_num[' Label'] = LabelEncoder().fit_transform(combined_all_num[' Label'])
    combined_all_num.drop('Flow Bytes/s', axis=1, inplace=True)

    combined_all_num = pd.DataFrame(StandardScaler().fit_transform(combined_all_num), columns=combined_all_num.columns)

    corr_matrix = combined_all_num.corr().abs()

    # Inicializar a estrutura de dados para o algoritmo de união e busca
    parent = {col: col for col in corr_matrix.columns}

    def find(col):
        if parent[col] != col:
            parent[col] = find(parent[col])
        return parent[col]

    def union(col1, col2):
        root1 = find(col1)
        root2 = find(col2)
        if root1 != root2:
            parent[root2] = root1

    # Iterar sobre a matriz de correlação para encontrar colunas correlacionadas
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and corr_matrix[col1][col2] > 0.9:
                union(col1, col2)

    # Agrupar colunas por seus representantes
    groups = {}
    for col in corr_matrix.columns:
        root = find(col)
        if root not in groups:
            groups[root] = []
        groups[root].append(col)

    # Converter os grupos em uma lista de listas
    column_groups_list = list(groups.values())

    # Exibir os conjuntos de colunas
    selected_columns = []
    for group in column_groups_list:
        selected_columns.append(group[0])
    return selected_columns

def select_cols_fast():
    return pd.read_csv("columns.csv").columns
    
def load_data(partition_id, batch_size):
    file = sorted(os.listdir("data"))[partition_id - 1]
    df = pd.read_csv(f"data/{file}")

    # replace inf values with 10^10
    df[' Flow Packets/s'] = df[' Flow Packets/s'].replace(np.inf, 10**10)
    df['Flow Bytes/s'] = df['Flow Bytes/s'].replace(np.inf, 10**10)

    # filter columns 
    selected_columns = select_cols_fast()
    df = df[selected_columns]

    # separate clean and fraud transactions
    clean = df[df[' Label'] == 'BENIGN']
    fraud = df[df[' Label'] != 'BENIGN']

    TRAINING_SAMPLE = int(0.85 * clean.shape[0])
    clean = clean.sample(frac=1).reset_index(drop=True)

    # training set: exlusively non-fraud transactions
    X_train = clean.iloc[:TRAINING_SAMPLE].drop(' Label', axis=1)

    # testing  set: the remaining non-fraud + all the fraud 
    X_test = clean.iloc[TRAINING_SAMPLE:]
    X_test = pd.concat([X_test, fraud])

    X_train, X_validate = train_test_split(X_train, 
                                        test_size=0.05, 
                                        random_state=1)

    # manually splitting the labels from the test df
    X_test, y_test = X_test.drop(' Label', axis=1).values, X_test[' Label'].values

    # configure our pipeline
    pipeline = Pipeline([('normalizer', Normalizer()),
                        ('scaler', MinMaxScaler())])
    
    pipeline.fit(X_train)

    X_train_transformed = pipeline.transform(X_train)
    X_validate_transformed = pipeline.transform(X_validate)
    X_test_transformed = pipeline.transform(X_test)

    X_train = torch.tensor(X_train_transformed, dtype=torch.float32)
    X_validate = torch.tensor(X_validate_transformed, dtype=torch.float32)
    X_test = torch.tensor(X_test_transformed, dtype=torch.float32)

    trainloader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
    validationloader = DataLoader(X_validate, batch_size=batch_size, drop_last=True)
    testloader = DataLoader(X_test, batch_size=batch_size)

    return trainloader, validationloader, testloader, y_test


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            data = batch.to(device)
           
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()


    val_loss= test(net, valloader, device)

    results = {
        "val_loss": val_loss,
    }
    return results


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)  
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            inputs = batch.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item() 
            
    avg_loss = total_loss / len(testloader) 
    print(f"Average MSE Loss: {avg_loss}")
    return avg_loss

def eval(net, testloader, y_test, device, thresholds, partition_id):
    """Validate the model on the test set."""
    net.eval()
    net.to(device)
    criterion = nn.MSELoss(reduction='none')  # Evita redução automática
    reconstructions = []

    with torch.no_grad():
        for batch in testloader:
            inputs = batch.to(device)

            # Faz a predição
            outputs = net(inputs)

            # Calcula o erro de reconstrução por amostra
            reconstruction_error = criterion(outputs, inputs).mean(dim=tuple(range(1, inputs.ndim)))  # Reduz nas dimensões exceto batch
            reconstructions.extend(reconstruction_error.cpu().numpy())

    

    # Gera um nome para o experimento baseado no timestamp
    plot_error(y_test, reconstructions, partition_id)
    

    # # Converte labels para 'BENIGN' ou 'FRAUD'
    # y_true = np.where(y_test == 'BENIGN', 'BENIGN', 'FRAUD')

    # # Testa diferentes thresholds
    # for th in thresholds:
    #     y_pred = np.where(np.array(reconstructions) > 10**(-th), 'FRAUD', 'BENIGN')
    #     plot_cm(y_pred, y_true, experiment_name, th)

    return float(np.mean(np.array(reconstructions)))

def plot_error(y_test, mse, partition_id):
    experiment = os.environ.get("EXPERIMENT_NAME", "default")
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("experiments/errors", exist_ok=True)
    os.makedirs(f"experiments/errors/{experiment}", exist_ok=True)
    path = f"experiments/errors/{experiment}/{partition_id}"
    os.makedirs(path, exist_ok=True)
    mse = pd.Series(mse)
    import matplotlib.pyplot as plt
    clean_error = mse[y_test=='BENIGN']
    fraud_error = mse[y_test!='BENIGN']

    fig, ax = plt.subplots(figsize=(6,6))

    ax.hist(-np.log10(clean_error), bins=50, density=True, label="clean", alpha=0.5, color="green")
    ax.hist(-np.log10(fraud_error), bins=50,   density=True,label="fraud", alpha=0.5, color="red")

    plt.title("(Normalized) Distribution of the Reconstruction Loss")
    plt.legend()
    
    #plt.savefig(f"{path}/error_{round_}.png", dpi=300, bbox_inches='tight', transparent=False)

def plot_cm(y_pred, y_true, experiment_name, threshold):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Criar diretório se não existir
    save_path = "experiments"
    os.makedirs(save_path, exist_ok=True)

    # Calcula a matriz de confusão
    class_names = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Configura a figura antes de plotar
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="inferno", cbar=False, 
                xticklabels=class_names, yticklabels=class_names)

    # Configurações do gráfico
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix (Percentage) - Threshold {threshold}', fontsize=15)

    # Salvar antes de exibir
    save_file = f"{save_path}/{experiment_name}_{threshold}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight', transparent=False)

