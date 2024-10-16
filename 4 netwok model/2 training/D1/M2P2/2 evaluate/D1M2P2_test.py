import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

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
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        x = x.unsqueeze(1).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
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

# Loss function
def custom_loss( y_pred, batch_y):
    w_loss = nn.MSELoss()(batch_y, y_pred)  # Mean square loss function, MSEW
    return w_loss


device = torch.device("cpu")

# Define hyperparameters
input_size = 5
output_size = 1
num_epochs = 880
batch_size = 32

hidden_size = 11
learning_rate = 0.000808669847370544
weight_decay = 0.00002185173691953

seed = 42
set_seed(seed)

data = pd.read_csv('experimental data D1M2P2.csv')  # Load the data

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

for epoch in range(num_epochs):
    final_model.train()
    epoch_loss = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        train_pred = final_model(batch_x[:, 5:])
        loss = custom_loss(train_pred, batch_y)
        final_optimizer.zero_grad()
        loss.backward()
        final_optimizer.step()
        epoch_loss += loss.item()
    average_epoch_loss = epoch_loss / len(train_loader)
    training_losses.append(average_epoch_loss)
    # Print epoch and corresponding loss
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_epoch_loss:.4f}')

    # Save training loss
    loss_df = pd.DataFrame(training_losses, columns=['Training Loss'])
    loss_df.to_csv('training_losses_D1M2P2.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Running time: {elapsed_time:.2f} s")

# Save model
torch.save(final_model.state_dict(), 'best M2P2 model_D1.pth')

final_model.eval()
with torch.no_grad():
    pre = scaler.inverse_transform(final_model(x_gen[:, 5:]).cpu().numpy())
    act = scaler.inverse_transform(y_gen.cpu().numpy())
    # Output predictions and actual values
    results_df = pd.DataFrame({
        'Actual': act.flatten(),
        'Predicted': pre.flatten()
    })
    results_df.to_csv('final_predictions_D1M2P2.csv', index=False)

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
    upper_bound = error_line * 1.25
    lower_bound = error_line * 0.85
    plt.fill_between(error_line, lower_bound, upper_bound, color='gray', alpha=0.3, label='20% Error')
    plt.legend()
    plt.show()

# Error calculation
relative_error = np.abs(act - pre) / act
f_mean_relative_error = np.mean(relative_error)
f_max_relative_error = np.max(relative_error)
print('f_Mean relative error: {:.4f}'.format(f_mean_relative_error))
print('f_Max relative error: {:.4f}'.format(f_max_relative_error))