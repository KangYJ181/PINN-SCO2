import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # the data normalization module
from sklearn.model_selection import KFold  # K-fold cross-validation module
from bayes_opt import BayesianOptimization
import time

start_time = time.time()

# Pseudo random, guaranteed repeatability
def set_seed(seed):
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed_all(seed)  # All GPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, nonlinearity='tanh', batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        x = x.unsqueeze(1).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])

        return out

    def initialize_weights(self):
        # Initializes the weights and biases of the RNN
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)  # Positive distribution initialization
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 0 initialization

        # Initialize the weight and bias of the fully connected layer
        for layer in [self.fc1]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0)

# Loss function
def custom_loss( y_pred, batch_y):
    w_loss = nn.MSELoss()(batch_y, y_pred)  # Mean square loss function, MSEW
    return w_loss

def train_and_evaluate(learning_rate, weight_decay, hidden_size):
    hidden_size = int(hidden_size)  # Convert hidden_size to int
    kf = KFold(n_splits=k_fold)  # K-fold validation
    error_ave = []
    for train_index, valid_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        valid_dataset = torch.utils.data.Subset(dataset, valid_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Initialize the network and optimizer
        model = RNN(input_size, hidden_size, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                train_pred = model(batch_x[:, 5:])
                loss = custom_loss(train_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        valid_preds = []
        valid_trues = []
        for i, (batch_x, batch_y) in enumerate(valid_loader):
            batch_x = batch_x.to(device)
            valid_pred = model(batch_x[:, 5:])
            valid_preds.append(valid_pred.cpu().numpy())
            valid_trues.append(batch_y.cpu().numpy())

        # Convert lists to arrays
        valid_preds = np.concatenate(valid_preds)
        valid_trues = np.concatenate(valid_trues)

        # Inverse normalization
        valid_preds = scaler.inverse_transform(valid_preds)
        valid_trues = scaler.inverse_transform(valid_trues)

        # Calculate the average error of each fold
        error = np.mean(np.abs((valid_preds - valid_trues) / valid_trues))
        error_ave.append(error)

    return np.mean(error_ave)


device = torch.device("cpu")

# Define hyperparameters
input_size = 5
output_size = 1
num_epochs = 880
batch_size = 32
k_fold = 10

seed = 42
set_seed(seed)

data = pd.read_csv('experimental data D2.csv')  # Load the data

# Save a copy of the original data
x_raw = data.drop('mas', axis=1).values
y_raw = data['mas'].values.reshape(-1, 1)

# Data preprocessing, scale the data to the 0~1 range
scaler = MinMaxScaler()
x = scaler.fit_transform(x_raw)
y = scaler.fit_transform(y_raw)

x = torch.from_numpy(np.hstack((x_raw, x))).float()
y = torch.from_numpy(y).float()

# Specify the training set
x = x[:229]
y = y[:229]

# Save the raw data of the training set
x_raw = x_raw[:229]
y_raw = y_raw[:229]

# Create a DataLoader object to pair input with output
dataset = torch.utils.data.TensorDataset(x, y)

# Bayesian optimization
pbounds = {'learning_rate': (0.0008, 0.0008), 'weight_decay': (4e-9, 4e-9), 'hidden_size':(14,14)}
optimizer = BayesianOptimization(f=train_and_evaluate, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=1, n_iter=0)
print(optimizer.max)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Running time: {elapsed_time:.2f} s")

# Record the optimization result
results = []
for res in optimizer.res:
    results.append({
        'hidden_size': res['params']['hidden_size'],
        'learning_rate': res['params']['learning_rate'],
        'weight_decay': res['params']['weight_decay'],
        'target': res['target']
    })

results_df = pd.DataFrame(results)
results_df.to_csv('optimization_results_DataDriven.csv', index=False)