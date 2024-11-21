from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class MyDataSet(Dataset):
    def __init__(self, input, output, time_step=200):
        self.time_step = time_step
        self.input = torch.tensor(input, dtype=torch.float32)
        self.ouput = torch.tensor(output, dtype=torch.float32)
        print(self.input.shape, self.ouput.shape)
        
    def __len__(self):
        return len(self.input) - self.time_step

    def __getitem__(self, index):
        feature = self.input[index:index+self.time_step, :]
        if self.ouput is not None:
            label = self.ouput[index+self.time_step-1]
            return feature, label
        else:
            return feature

class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size) # 全连接层
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.linear1(x)
        return x[:, -1, :]

def train(net, train_iter, loss, epochs, lr):
    net.to("cuda")
    trainer = torch.optim.Adam(net.parameters(), lr)
    loss_d = []
    for epoch in range(epochs):
        for X, y in train_iter:
            # print(X.shape, y.shape)
            X, y = X.to("cuda"), y.to("cuda")
            trainer.zero_grad()
            predic = net(X)
            # print(predic.shape, y.shape)
            l = loss(predic, y)
            l.sum().backward()
            trainer.step()
            loss_d.append(l.sum().item())
        if (epoch+1)%100==0:
            print("step: {0} , loss: {1}".format(epoch+1,l.sum().item()))
        # early_stopping(l.sum().item())
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    return loss

full_data = pd.read_csv("climate.csv")
full_rows_num = len(full_data)

p_col = full_data["p (mbar)"].to_numpy().reshape(full_rows_num, -1)
t_col = full_data["T (degC)"].to_numpy().reshape(full_rows_num, -1)
tpot_col = full_data["Tpot (K)"].to_numpy().reshape(full_rows_num, -1)
tdew_col = full_data["Tdew (degC)"].to_numpy().reshape(full_rows_num, -1)
rh_col = full_data["rh (%)"].to_numpy().reshape(full_rows_num, -1)
vpmax_col = full_data["VPmax (mbar)"].to_numpy().reshape(full_rows_num, -1)
vpact_col = full_data["VPact (mbar)"].to_numpy().reshape(full_rows_num, -1)
sh_col = full_data["sh (g/kg)"].to_numpy().reshape(full_rows_num, -1)
h2oc_col = full_data["H2OC (mmol/mol)"].to_numpy().reshape(full_rows_num, -1)
rho_col = full_data["rho (g/m**3)"].to_numpy().reshape(full_rows_num, -1)
wv_col = full_data["wv (m/s)"].to_numpy().reshape(full_rows_num, -1)
maxwv_col = full_data["max. wv (m/s)"].to_numpy().reshape(full_rows_num, -1)
wd_col = full_data["wd (deg)"].to_numpy().reshape(full_rows_num, -1)

input = np.column_stack((p_col,tpot_col,tpot_col,tdew_col,rh_col,vpmax_col,vpact_col,sh_col,h2oc_col,rho_col,wv_col,maxwv_col,wd_col))
output = t_col

input = input[:30000]
output = output[:30000]
dataset = MyDataSet(input, output, 500)
print(dataset[0][0].shape)

train_dataloader=DataLoader(dataset=dataset,batch_size=2000,shuffle=True,num_workers=6)
early_stopping = EarlyStopping(patience=10)

num_epochs = 3000
loss = nn.MSELoss()
net = LstmRNN(13, 16, output_size=1, num_layers=1)
print('LSTM model:', net)
loss_function = nn.MSELoss()

loss_d = train(net, train_dataloader, loss, num_epochs, 0.001)
plt.subplot(311)  
plt.plot(loss_d)
plt.title("loss")
# torch.save(net, 'net1.pth')

input = input[2000:10000]
output = output[2000:10000]
dataset1 = MyDataSet(input, output)

dataloader1=DataLoader(dataset=dataset1,batch_size=2000,shuffle=False,num_workers=6)

model = torch.load("net1.pth")
model.eval()
model.to("cuda")
for x1, y1 in dataloader1:
    x1, y1 = x1.to("cuda"), y1.to("cuda")
    predic1 = model(x1)

plt.subplot(313)
plt.scatter(range(len(y1)), y1.cpu().detach().numpy(), color="b", s=5, label="test_data")
plt.plot(range(len(predic1)), predic1.cpu().detach().numpy())

plt.show()
