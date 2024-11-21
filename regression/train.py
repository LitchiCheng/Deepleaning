from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
import csv

data_num = 3000
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['y','x'])

    # x=np.linspace(-6*np.pi,6*np.pi,data_num)
    # y=np.sin(x) + np.random.normal(loc=0.03, scale=0.1, size=(data_num))

    x=np.linspace(-6*np.pi,6*np.pi,data_num)
    y=np.sin(x)/x
    y=y+np.random.normal(loc=0.03, scale=0.1, size=(data_num))

    for xi,yi in zip(x,y):
        csvwriter.writerow([yi,xi])

class MyDataSet(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, nrows=3000)
        self.x = df["x"].to_numpy()
        self.x = np.expand_dims(self.x,axis=1)
        self.y = torch.tensor(df['y'].to_numpy().reshape(3000, -1),dtype=torch.float)
        self.x = torch.tensor(self.x,dtype=torch.float)
        print(self.x.shape, self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

data_set = MyDataSet('output.csv')
data_loader = DataLoader(dataset=data_set, batch_size=500, shuffle=True, drop_last=False)

def train(net, train_iter, loss, epochs, lr):
    net.to("cuda")
    trainer = torch.optim.Adam(net.parameters(), lr)
    loss_d = []
    for epoch in range(epochs):
        for X, y in train_iter:
            X, y = X.to("cuda"), y.to("cuda")
            trainer.zero_grad()
            predic = net(X)
            l = loss(predic, y)
            l.sum().backward()
            trainer.step()
            loss_d.append(l.sum().item())
        if (epoch+1)%100==0:
            print("step: {0} , loss: {1}".format(epoch+1,l.sum().item()))
    plt.subplot(311)  
    plt.plot(loss_d)
    plt.title("loss")

num_epochs = 2000
loss = nn.MSELoss()
net = nn.Sequential(
            nn.Linear(1,10),nn.ReLU(),
            nn.Linear(10,128),nn.ReLU(),
            nn.Linear(128,10),nn.ReLU(),
            nn.Linear(10,1)
        )
train(net, data_loader, loss, num_epochs, 0.001)

if True:
    df2 = pd.read_csv("output.csv")  
    full_x = df2["x"]
    full_y = df2["y"]
    print(len(full_x), len(full_y))
    plt.subplot(312)
    plt.scatter(full_x, full_y, color="r")
    plt.title("train_data")

df3 = pd.read_csv("output.csv",skiprows=500,names=['y', 'x'])
testx = df3["x"]
testy = df3["y"]
plt.subplot(313)
plt.scatter(testx, testy, color="b", s=5, label="test_data")

testx_tensor = np.expand_dims(testx, axis=1)
testx_tensor = torch.tensor(testx_tensor, dtype=torch.float)
predicty = net(testx_tensor.to("cuda"))
plt.subplot(313)
plt.plot(testx, predicty.cpu().detach().numpy(),"r-", label="predict_curve")
plt.legend()

plt.show()



