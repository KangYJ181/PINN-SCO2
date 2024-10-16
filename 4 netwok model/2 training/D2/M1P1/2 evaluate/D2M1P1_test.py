import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import matplotlib.pyplot as plt
import time



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
        self.d = nn.Parameter(torch.randn(1).to(device))
        self.e = nn.Parameter(torch.randn(1).to(device))
        self.f = nn.Parameter(torch.randn(1).to(device))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, nonlinearity='tanh', batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size - 6)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        x = x.unsqueeze(1).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        coefficients = torch.stack([self.a, self.b, self.c, self.d, self.e, self.f]).view(1, -1)
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

    def physics(self, ldr, pre, dia, tem):
        # Edit the physical constraint P2
        # ldr: length to diameter ratio
        # pre: pressure (MPa)
        # dia: diameter (mm)
        # tem: temperature (K)
        # rou: density (kg/m3)
        # w: mass flow rate (g/s)
        P = pre * 1000000  # Convert the pressure unit to Pa so that REFPROP can use it
        z = [1.0]  # For pure fluid, to call density for REFPROP
        result = RP.REFPROPdll('CO2', 'PT', 'D', RP.MASS_BASE_SI, iMass, iFlag, P, tem, z)
        rou = result.Output[0]
        w = torch.pi * (dia / 2) ** 2 \
            * (self.a + self.b * (pre / 7.39) ** self.c * (rou / 354.36) ** self.d) \
            * (2 * rou * pre * (1 - (self.e + self.f * torch.log(ldr)))) ** 0.5
        return w


def physics_constraints(model, batch_x):
    physics_outputs = []
    for x in torch.split(batch_x, 1):
        # Acquisition of physical quantity
        ldr = x[:, 0]
        pre = x[:, 1]
        dia = x[:, 2]
        tem = x[:, 4]
        w = model.physics(ldr, pre, dia, tem)
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


device = torch.device("cpu")

# Define hyperparameters
input_size = 5
output_size = 1
batch_size = 32
num_epochs = 880

learning_rate = 0.000996593824273852
w_loss_weight1 = 0.0290459434034389
w_loss_weight2 = 0.0778196342188175
weight_decay = 0.0000544796856125476
hidden_size = 9

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

# Specify the training set and validation set
x_gen = x[0:]
y_gen = y[0:]
x = x[:229]
y = y[:229]

# Save the raw data of the training set
x_raw = x_raw[:229]
y_raw = y_raw[:229]

# Create a DataLoader object to pair input with output
dataset = torch.utils.data.TensorDataset(x, y)


# Retrain the model using the best parameters and the entire data set
training_losses = []
final_model = RNN(input_size, hidden_size, output_size).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize the coefficients according to the reference paper
final_model.a.data.fill_(0.5463)
final_model.b.data.fill_(0.0587)
final_model.c.data.fill_(2.07)
final_model.d.data.fill_(-0.939)
final_model.e.data.fill_(0.579)
final_model.f.data.fill_(0.024)

for epoch in range(num_epochs):
    final_model.train()
    epoch_loss = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        train_pred = final_model(batch_x[:, 5:])
        loss = custom_loss(final_model, train_pred, batch_y, batch_x, w_loss_weight1, w_loss_weight2)
        final_optimizer.zero_grad()
        loss.backward()
        final_optimizer.step()
        epoch_loss += loss.item()

        # Limit the range of coefficient variation
        with torch.no_grad():
            final_model.a.data.clamp_(min=0.5, max=0.55)
            final_model.b.data.clamp_(min=0.03, max=0.06)
            final_model.c.data.clamp_(min=1.5, max=2.1)
            final_model.d.data.clamp_(min=-1.0, max=-0.5)
            final_model.e.data.clamp_(min=0.4, max=0.6)
            final_model.f.data.clamp_(min=0.01, max=0.05)

    average_epoch_loss = epoch_loss / len(train_loader)
    training_losses.append(average_epoch_loss)
    # Print epoch and corresponding loss
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_epoch_loss:.4f}')

    # Save training loss
    loss_df = pd.DataFrame(training_losses, columns=['Training Loss'])
    loss_df.to_csv('training_losses_D2M1P1.csv', index=False)



# Save model
torch.save(final_model.state_dict(), 'best M1P1 model_D2.pth')

# Save coefficients
coefficients = {
    'a': final_model.a.item(),
    'b': final_model.b.item(),
    'c': final_model.c.item(),
    'd': final_model.d.item(),
    'e': final_model.e.item(),
    'f': final_model.f.item()
}
coefficients_df = pd.DataFrame([coefficients])
coefficients_df.to_csv('optimized_coefficients_D2M1P1.csv', index=False)

final_model.eval()
with torch.no_grad():

    start_time = time.time()

    pre = scaler.inverse_transform(final_model(x_gen[:, 5:]).cpu().numpy())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time:.5f} s")

    act = scaler.inverse_transform(y_gen.cpu().numpy())
    # Output predictions and actual values
    results_df = pd.DataFrame({
        'Actual': act.flatten(),
        'Predicted': pre.flatten()
    })
    results_df.to_csv('final_predictions_D2M1P1.csv', index=False)

    plt.plot(act, label='Actual')
    plt.plot(pre, label='Predicted')
    plt.legend()
    plt.show()

    relative_error = np.abs(act - pre) / act
    plt.plot(relative_error, label='Relative Error')
    plt.legend()
    plt.show()

    plt.scatter(act, pre)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Relative Error Distribution')
    plt.plot([np.min(act), np.max(act)], [np.min(act), np.max(act)], color='red', label='y=x')

    # Add the 20% error line
    error_line = np.linspace(np.min(act), np.max(act), 100)
    upper_bound = error_line * 1.15
    lower_bound = error_line * 0.7
    plt.fill_between(error_line, lower_bound, upper_bound, color='gray', alpha=0.3, label='20% Error')
    plt.legend()
    plt.show()

# Error calculation
relative_error = np.abs(act - pre) / act
f_mean_relative_error = np.mean(relative_error)
f_max_relative_error = np.max(relative_error)
print('f_Mean relative error: {:.4f}'.format(f_mean_relative_error))
print('f_Max relative error: {:.4f}'.format(f_max_relative_error))