import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import time
import pickle
import os

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
        self.a = nn.Parameter(torch.randn(1).to(device))
        self.b = nn.Parameter(torch.randn(1).to(device))
        self.c = nn.Parameter(torch.randn(1).to(device))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, nonlinearity='tanh', batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size - 3)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        x = x.unsqueeze(1).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        coefficients = torch.stack([self.a, self.b, self.c]).view(1, -1)
        coefficients = coefficients.repeat(out.size(0), 1)
        out = torch.cat((out, coefficients), dim=1)
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        # Initializes the weights and biases of the RNN
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)  # Positive distribution initialization
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # 0 initialization

        # Initialize the weight and bias of the fully connected layer
        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def physics(self, ldr, pre, dia, miu, tem):
        # Edit the physical constraint P2
        # ldr: length to diameter ratio
        # pre: pressure (MPa)
        # miu: roughness (μm)
        # dia: diameter (mm)
        # tem: temperature (K)
        # rou: density (kg/m3)
        # w: mass flow rate (g/s)
        P = pre * 1000000  # Convert the pressure unit to Pa so that REFPROP can use it
        z = [1.0]  # For pure fluid, to call density for REFPROP
        result = RP.REFPROPdll('CO2', 'PT', 'D', RP.MASS_BASE_SI, iMass, iFlag, P, tem, z)
        rou = result.Output[0]
        w = torch.pi * ((dia / 2)) ** 2 * (2 * (pre - 7.39) / ((1 / rou) * (self.a + ((self.b * torch.log10(miu * 1e-3 / dia) - self.c) ** (-2)) * ldr + 1))) ** 0.5

        return w


def physics_constraints(model, batch_x):
    physics_outputs = []
    for x in torch.split(batch_x, 1):
        # Acquisition of physical quantity
        ldr = x[:, 0]
        pre = x[:, 1]
        dia = x[:, 2]
        miu = x[:, 3]
        tem = x[:, 4]
        w = model.physics(ldr, pre, dia, miu, tem)
        physics_outputs.append(w)
    return torch.stack(physics_outputs).squeeze()


# Loss function with physical constraint P2
def custom_loss(model, y_pred, y_true, batch_x, w_loss_weight1, w_loss_weight2):
    mse_loss = nn.MSELoss()(y_pred, y_true)  # Mean square loss function, MSEW
    x_raw = batch_x[:, :5]
    w = physics_constraints(model, x_raw)
    w_numpy = w.detach().cpu().numpy().reshape(-1, 1)
    w_normalized = scaler.transform(w_numpy)
    w_tensor = torch.from_numpy(w_normalized).float().to(batch_x.device)
    w_loss = nn.MSELoss()(w_tensor, y_pred)  # Mean square error between the predicted value of y and w, MSER
    w_loss_ = nn.MSELoss()(w_tensor, y_true)   # Mean square error between the actual value of y and w, MSET
    total_loss = (1 - w_loss_weight1 - w_loss_weight2) * mse_loss + w_loss_weight1 * w_loss + w_loss_weight2 * w_loss_  # The total loss function
    return total_loss


def train_and_evaluate(w_loss_weight1, w_loss_weight2, learning_rate, weight_decay, hidden_size):
    hidden_size = int(hidden_size)
    # Check if weights are valid
    if (w_loss_weight1 + w_loss_weight2) > 0.5 or (w_loss_weight1 > w_loss_weight2):
        return 1e-6  # Return a large negative value to discourage this choice

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

        # Initialize the coefficients according to the reference paper
        model.a.data.fill_(0.4887)
        model.b.data.fill_(1.07148)
        model.c.data.fill_(2.07908)

        for epoch in range(num_epochs):
            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                train_pred = model(batch_x[:, 5:])
                loss = custom_loss(model, train_pred, batch_y, batch_x, w_loss_weight1, w_loss_weight2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Limit the range of coefficient variation
                with torch.no_grad():
                    model.a.data.clamp_(min=0.25, max=0.75)
                    model.b.data.clamp_(min=0.5, max=1.5)
                    model.c.data.clamp_(min=1.0, max=3.0)

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

return 1 - np.mean(error_ave)


device = torch.device("cpu")

# Define hyperparameters
input_size = 5
output_size = 1
num_epochs = 880
batch_size = 32
k_fold = 10

seed = 42
set_seed(seed)

# Set the REFPROP
RP = REFPROPFunctionLibrary('C:/Program Files (x86)/REFPROP')
iMass = 1  # 1 represents the mass basis
iFlag = 0  # 0 represents the standard calculation process
MASS_BASE_SI = RP.GETENUMdll(iFlag, "MASS BASE SI").iEnum  # Only REFPROP 10.0 can use this function

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
pbounds = {'w_loss_weight1': (0.01, 0.49), 'w_loss_weight2': (0.01, 0.49), 'learning_rate': (0.0005,0.001), 'weight_decay': (1e-5,1e-4), 'hidden_size': (7,12)}

# Load optimizer state if it exists
if os.path.exists('optimizer_state_D2M1P2.pkl'):
    with open('optimizer_state_D2M1P2.pkl', 'rb') as f:
        optimizer = pickle.load(f)
else:

    optimizer = BayesianOptimization(f=train_and_evaluate, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=46, n_iter=90)  # Ensure 10 valid initial sampling points and 70 valid iterations

# Save optimizer state
with open('optimizer_state_D2M1P2.pkl', 'wb') as f:
    pickle.dump(optimizer, f)

print(optimizer.max)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Running time: {elapsed_time:.2f} s")

# Record the optimization result
results = []
for res in optimizer.res:
    results.append({
        'w_loss_weight1': res['params']['w_loss_weight1'],
        'w_loss_weight2': res['params']['w_loss_weight2'],
        'learning_rate': res['params']['learning_rate'],
        'weight_decay': res['params']['weight_decay'],
        'hidden_size': res['params']['hidden_size'],
        'target': res['target']
    })

results_df = pd.DataFrame(results)
results_df.to_csv('optimization_results_D2M1P2.csv', index=False)
