# Instrution Detection using federated autoencoders

Personal project to learn better how anomaly detection with federated autoencoders works

### Tools
- Torch
- Flower
- Scikit-Learn

### Dataset
- [CIC-IDS-2017 dataset ](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

### How to run (training and eval)
```bash
flwr run .
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/vinic011/instrusion-fl.git
```
2. Navigate to the project directory:
```bash
cd instrusion-fl
```
3. Install the required dependencies:
```bash
pip install -e .
```

### Data Preparation

1. Dropped columns with just one value
2. Replaced inf values by a big one, but not inf
3. Groupped high correlation features (> 0.9 ), just selected one for group
4. Normalized based on training set

### How does Anomaly detection with autoencoders work?
1. Autoencoder train and validation data should be just clean (no anomaly)
2. At test, we must have clean and anomaly data
3. We will compute the reconstruction error of test data, i.e. MSE(y_pred - y_true)
4. Based on this error, we must set a threshold that separates clean and anomaly data

### Architecture Selection
For tabular data, we decided using simple Feed Foward NN.

We decreased the architecture size from the INPUT_DIM = 39 to 4, that based on supervised learning, must be more or less the problem dimensionality. We confirmed this by setting bigger bottenecks and not having smaller reconstructions errors for clean data. Smaller bottlenecks were not able to distinguish well clean and fraud data.

### Comparing federated x centralized
